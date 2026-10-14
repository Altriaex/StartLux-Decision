"""The transformers engine on a tiny random model with the architecture of a real folder (CPU): the bidirectional
prompt read-out of Instant folders (the checks of bidirectional.enable on the model itself, and that the read-out
really differs from the causal one), the causal read-out against the stock forward, generation, and the pipeline.

Needs torch, transformers 5.8 and STARTLUX_TEST_MODEL_DIR (a model folder: only its config.json is read, for the
architecture; the weights are random). The test runs on CPU. Where flash-linear-attention / causal-conv1d are
installed, transformers would call their CUDA kernels on CPU tensors, so the test switches the model module to its
PyTorch paths while it runs."""
from __future__ import annotations

import asyncio
import gc
import importlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from startlux_decide.backends.base import Sampling
from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.controls import Controls
from startlux_decide.modelcfg import load_model_folder
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import from_systemone, messages, plain_prompt

MODEL_DIR = os.environ.get("STARTLUX_TEST_MODEL_DIR")
try:
    import torch
    import transformers

    from startlux_decide.backends.transformers import TransformersBackend
    READY = transformers.__version__.startswith("5.8.") and bool(MODEL_DIR)
except ImportError:
    READY = False

VOCAB = 2048
FAST_PATHS = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
              "FusedRMSNormGated")


class SmallTokenizer(FakeTokenizer):
    """FakeTokenizer with ids that fit a 2,048-token vocabulary (ASCII text only)."""
    eos_token_id, pad_token_id = 2001, 0

    def encode(self, text, add_special_tokens=False):
        return [t - 2_000_000 + 2000 if t >= 2_000_000 else t % 1900 + 100 for t in super().encode(text)]

    def decode(self, ids, skip_special_tokens=False):
        return super().decode([t - 2000 + 2_000_000 if t >= 2000 else (t - 100) % 1900 for t in ids],
                              skip_special_tokens)

    def convert_tokens_to_ids(self, token):
        return self.encode(token)[0]


def model_module():
    """The transformers module of the folder's architecture (its fast paths are switched off on CPU)."""
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        kind = json.load(f)["model_type"]
    return importlib.import_module(f"transformers.models.{kind}.modeling_{kind}")


def tiny_folder(bidirectional: bool) -> str:
    """A random 8-layer model with the folder's text architecture (full attention in every 4th layer, as the real
    ones). Two full-attention layers are needed: the last position reads every position in either mode, so a
    bidirectional layer changes the read-out only through the layers after it."""
    config_class = type(transformers.AutoConfig.from_pretrained(MODEL_DIR).get_text_config())
    cfg = config_class(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4,
                       linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
                       linear_num_value_heads=4, tie_word_embeddings=True, eos_token_id=2001, pad_token_id=0,
                       layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2)
    torch.manual_seed(0)
    path = tempfile.mkdtemp(prefix="startlux_tiny_")
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    dec = {"letter_token_ids": SmallTokenizer().encode("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
           "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0}}
    if bidirectional:
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        c["jev_prompt_attention"] = "bidirectional"
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(c, f)
        dec["prompt_attention"] = "bidirectional"
    with open(os.path.join(path, "decision_config.json"), "w") as f:
        json.dump(dec, f)
    return path


@unittest.skipUnless(READY, "needs torch, transformers 5.8 and STARTLUX_TEST_MODEL_DIR")
class TransformersEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = model_module()
        fast = {n: None for n in FAST_PATHS if getattr(module, n, None) is not None}
        cls.patch = mock.patch.multiple(module, **fast) if fast else None
        if cls.patch:
            cls.patch.start()
        cls.causal, cls.instant = tiny_folder(False), tiny_folder(True)
        cls.tok = SmallTokenizer()
        cls.engines = {}
        for name, path, bi in (("causal", cls.causal, False), ("instant", cls.instant, True)):
            e = TransformersBackend(path, cls.tok, bi, device="cpu", dtype="float32")
            asyncio.run(e.start())
            cls.engines[name] = e
        msgs = messages(from_systemone({"ticket": "charged twice for order 4411"},
                                       {"type": "choice", "instructions": "Which team?",
                                        "criteria": {"billing": "Payments", "shipping": "Delivery", "tech": None}}))
        cls.prompt = plain_prompt(cls.tok, msgs)

    @classmethod
    def tearDownClass(cls):
        for e in cls.engines.values():
            asyncio.run(e.close())
        cls.engines.clear()                 # release the weight files before removing the folders
        gc.collect()
        if cls.patch:
            cls.patch.stop()
        shutil.rmtree(cls.causal, ignore_errors=True)
        shutil.rmtree(cls.instant, ignore_errors=True)

    def test_bidirectional_checks_pass_on_the_model(self):
        report = self.engines["instant"].describe()["bidirectional_check"]
        self.assertEqual(report["full_attention_layers"], [3, 7])
        self.assertLess(report["causal_loop_max_diff"], 1e-4)
        self.assertEqual(report["padding_invariance_max_diff"], 0.0)
        self.assertGreater(report["late_token_reach"], 0.0)
        self.assertIsNone(self.engines["causal"].describe()["bidirectional_check"])

    def test_readouts(self):
        letters = self.tok.encode("ABC")
        causal = asyncio.run(self.engines["causal"].readout(self.prompt, letters))
        bidir = asyncio.run(self.engines["instant"].readout(self.prompt, letters))
        model = self.engines["causal"].model
        with torch.no_grad():
            stock = model(input_ids=torch.tensor([self.tok.encode(self.prompt)])).logits[0, -1, letters].tolist()
        for a, b in zip(causal, stock):
            self.assertAlmostEqual(a, b, places=4)
        self.assertGreater(max(abs(a - b) for a, b in zip(causal, bidir)), 1e-4)   # same weights, other attention

    def test_generation(self):
        thinking = self.prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        g = asyncio.run(self.engines["causal"].generate(thinking, 12, Sampling(0.6, 0.95, 20), ["</think>"], 1234))
        self.assertLessEqual(g.tokens, 12)
        self.assertIn(g.finish_reason, ("stop", "length"))
        with self.assertRaises(NotImplementedError):
            asyncio.run(self.engines["instant"].generate(self.prompt, 4, Sampling(0.6, 0.95, 20), [], 1))

    def test_pipeline(self):
        q = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}},
             "late": {"type": "noul", "instructions": "Late?"}}
        for name, path in (("causal", self.causal), ("instant", self.instant)):
            d = Decider(load_model_folder(path), self.engines[name])
            mode = "always" if name == "causal" else "never"
            out = asyncio.run(d.decide("order 4411", q, Controls(think_mode=mode, budget=16, details=True)))
            self.assertEqual(set(out.answers), {"team", "late"})
            self.assertAlmostEqual(sum(out.answers["team"]["probabilities"].values()), 1.0)
            self.assertEqual(out.answers["late"]["details"]["thought"]["used"], name == "causal")


if __name__ == "__main__":
    unittest.main()
