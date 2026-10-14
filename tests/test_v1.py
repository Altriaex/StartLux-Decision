"""v1 folders (decision_config.json "format": "StartLux-Decision-v1", no auto_think block): causal, one pass by
default, thinking only when a request asks; the prompts of v1.1; nothing on the vLLM path refuses them (the launcher,
the dense and the MoE architecture, the plugin stays inactive)."""
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import unittest
from types import SimpleNamespace

from startlux_decide import vllm_plugin as P
from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls
from startlux_decide.launch import VLLMServers
from startlux_decide.modelcfg import V1_FORMAT, load_model_folder
from startlux_decide.pipeline import Decider, default_controls

from .folders import make_folder
from .test_golden import random_question, random_state
from .test_vllm_plugin import _PluginState

ARCH = {"dense": {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"},
        "moe": {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}}


def v1_folder(arch: str = "dense", extra: dict | None = None) -> str:
    """A folder as the v1 release ships it: format, the four decision keys, no prompt_attention, no auto_think."""
    return make_folder(extra={"format": V1_FORMAT, **(extra or {})}, config=ARCH[arch])


def run(coro):
    return asyncio.run(coro)


class V1FolderTest(unittest.TestCase):
    def setUp(self):
        self.dense, self.moe, self.v11 = v1_folder(), v1_folder("moe"), make_folder(config=ARCH["dense"])

    def tearDown(self):
        for f in (self.dense, self.moe, self.v11):
            shutil.rmtree(f, ignore_errors=True)

    def test_loads_as_causal_one_pass(self):
        for path in (self.dense, self.moe):
            folder = load_model_folder(path)
            self.assertEqual((folder.prompt_attention, folder.kind, folder.format, folder.v1),
                             ("causal", "v1", V1_FORMAT, True))
            self.assertEqual(folder.auto_think.think_mode, "never")
            self.assertEqual(default_controls(folder).think_mode, "never")
            self.assertFalse(folder.bidirectional)
        v11 = load_model_folder(self.v11)
        self.assertEqual((v11.kind, v11.auto_think.think_mode, v11.v1), ("auto_think", "auto", False))

    def test_an_auto_think_block_makes_it_autothink(self):
        path = v1_folder(extra={"auto_think": {"gate": 0.8}})
        try:
            folder = load_model_folder(path)
            self.assertEqual((folder.kind, folder.auto_think.think_mode, folder.auto_think.gate),
                             ("auto_think", "auto", 0.8))
        finally:
            shutil.rmtree(path)

    def test_one_pass_by_default_thinking_on_request(self):
        d = Decider(load_model_folder(self.dense), FakeBackend())
        q = {"q": {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(6)}}}
        out = run(d.decide("x", q, d.controls({"return_details": True})))
        self.assertFalse(out.answers["q"]["details"]["thought"]["used"])          # even when the top is low
        self.assertLess(out.answers["q"]["details"]["first_pass"]["top"], 0.6)
        self.assertFalse(any(c[0].startswith("generate") for c in d.backend.calls))
        for mode in ("always", "auto"):                                          # allowed when asked
            ctl = d.controls({"think_mode": mode, "gate": 1.0, "return_details": True})
            out = run(d.decide("x", q, ctl))
            self.assertTrue(out.answers["q"]["details"]["thought"]["used"])

    def test_prompts_are_those_of_v11(self):
        """The same requests give the same engine calls (prompts, letters, thought requests) on v1 and v1.1."""
        r = random.Random(11)
        for case in range(200):
            state = random_state(r)
            questions = {f"q{i}": random_question(r) for i in range(r.choice([1, 2, 3]))}
            ctl = Controls(think_mode=r.choice(["never", "always"]), details=True)
            calls = []
            for path in (self.dense, self.v11):
                backend = FakeBackend()
                out = run(Decider(load_model_folder(path), backend).decide(state, questions, ctl))
                calls.append((sorted(map(repr, backend.calls)), {k: {x: y for x, y in a.items() if x != "details"}
                                                                 for k, a in out.answers.items()}))
            self.assertEqual(calls[0], calls[1], case)


class V1OnVLLMTest(unittest.TestCase):
    def setUp(self):
        self.folders = [v1_folder(), v1_folder("moe")]

    def tearDown(self):
        for f in self.folders:
            shutil.rmtree(f, ignore_errors=True)

    def test_launcher_treats_v1_as_causal(self):
        for path in self.folders:
            for version in ("0.19.1", "0.31.0"):
                s = VLLMServers(path, ["0"], python="py", vllm_version=version)
                cmd, env = s.plan()[0]
                self.assertFalse(s.bidirectional)
                self.assertNotIn(P.ENV, env)                                # the plugin is not switched on
                self.assertEqual("--enable-prefix-caching" in cmd, version == "0.31.0")
                self.assertNotIn("--speculative-config", cmd)               # no mtp.* tensors in these folders

    def test_plugin_stays_inactive(self):
        """Even with STARTLUX_PREFIX_BIDIR=1 and the 0.31 settings of causal folders, a v1 runner reads causally."""
        for path in self.folders:
            with _PluginState():
                P.STATUS["installed"] = True
                with open(os.path.join(path, "config.json")) as f:
                    cfg = json.load(f)
                runner = SimpleNamespace(
                    model_config=SimpleNamespace(model=path, tokenizer=path, hf_config=SimpleNamespace(**cfg),
                                                 max_model_len=65536, is_encoder_decoder=False),
                    scheduler_config=SimpleNamespace(enable_chunked_prefill=True, long_prefill_token_threshold=0,
                                                     max_num_batched_tokens=8192),
                    cache_config=SimpleNamespace(enable_prefix_caching=True, cache_dtype="auto"),
                    parallel_config=SimpleNamespace(), speculative_config=object(), device="cpu")
                st = P._runner_init(runner, "V2")
                self.assertFalse(st.bi)
                self.assertEqual(st.mode, "causal")
                self.assertEqual(P.STATUS["bi_runners"], 0)


if __name__ == "__main__":
    unittest.main()
