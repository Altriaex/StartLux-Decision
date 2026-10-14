"""The MLX engine (startlux_decide/backends/mlx.py): the pure helpers always; the model path on a tiny random
Qwen3.5-style model when mlx-vlm is installed (Apple Silicon); a real folder when STARTLUX_MLX_FOLDER names one."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:          # only the mlx-vlm tests use numpy, and mlx-vlm brings it
    np = None

from startlux_decide.backends import mlx as M
from startlux_decide.backends.base import Sampling
from startlux_decide.images import IMAGE_PAD, load_media

HAVE_MLX = importlib.util.find_spec("mlx_vlm") is not None
IMAGE_ID = 250


def run(coro):
    return asyncio.run(coro)


@unittest.skipIf(np is None, "numpy is not installed")
class RopeIndexTest(unittest.TestCase):
    def test_text_only(self):
        pos, delta = M.rope_index([5, 6, 7, 8], IMAGE_ID, [])
        self.assertEqual(pos.tolist(), [[0, 1, 2, 3]] * 3)
        self.assertEqual(delta, 0)

    def test_one_image(self):
        ids = [5, 6, 252] + [IMAGE_ID] * 6 + [253, 7, 8]            # image of 4 x 6 patches = 2 x 3 merged
        pos, delta = M.rope_index(ids, IMAGE_ID, [[1, 4, 6]])
        self.assertEqual(pos[0].tolist(), [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8])
        self.assertEqual(pos[1].tolist(), [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8])
        self.assertEqual(pos[2].tolist(), [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8])
        self.assertEqual(delta, 9 - 12)

    def test_two_images(self):
        ids = [252] + [IMAGE_ID] * 4 + [253, 9, 252] + [IMAGE_ID] * 2 + [253]
        pos, delta = M.rope_index(ids, IMAGE_ID, [[1, 4, 4], [1, 2, 4]])
        self.assertEqual(pos[0].tolist(), [0, 1, 1, 1, 1, 3, 4, 5, 6, 6, 8])
        self.assertEqual(pos[1].tolist(), [0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 8])
        self.assertEqual(pos[2].tolist(), [0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(delta, 9 - 11)

    def test_mismatches(self):
        with self.assertRaises(ValueError):
            M.rope_index([252, IMAGE_ID, 253], IMAGE_ID, [[1, 4, 4]])       # 1 token, the grid needs 4
        with self.assertRaises(ValueError):
            M.rope_index([252, IMAGE_ID, 253], IMAGE_ID, [])                  # an image without a grid
        with self.assertRaises(ValueError):
            M.rope_index([5, 6], IMAGE_ID, [[1, 2, 2]])                       # a grid without an image


class TextHelpersTest(unittest.TestCase):
    def test_first_stop(self):
        self.assertIsNone(M.first_stop("abc</th", ["</think>"], 2))
        self.assertEqual(M.first_stop("abc</think>", ["</think>"], 3), 3)
        self.assertEqual(M.first_stop("abc</think>", ["zz", "</think>"], 11), 3)

    def test_detokenizer_waits_for_complete_characters(self):
        class Tok:
            def decode(self, ids, skip_special_tokens=True):
                return "�" if ids == [1] else "".join(chr(96 + i) for i in ids)
        d = M._Detokenizer(Tok())
        self.assertEqual(d.add(1), "")
        self.assertEqual(d.add(2), "ab")
        self.assertEqual(d.flush(), "")


@unittest.skipUnless(HAVE_MLX, "mlx-vlm is not installed")
class SamplerTest(unittest.TestCase):
    def test_sampling_rules(self):
        import mlx.core as mx
        logits = mx.array([[0.0, 5.0, 1.0, 2.0]])
        one = lambda u, **k: int(M.sample_tokens(logits, mx.array([u]), **{"temperature": 1.0, "top_k": 0, "top_p": 1.0, **k})[0])
        self.assertEqual(one(0.5, temperature=0.0), 1)                       # greedy
        self.assertEqual(one(0.999, top_k=1), 1)                             # top-1 only
        self.assertEqual(one(0.999, top_k=2), 3)                             # the 2nd of the 2 kept
        self.assertEqual(one(0.999, top_p=0.5), 1)                           # the top token alone reaches 0.5
        self.assertEqual(one(0.0), 1)                                        # u = 0 takes the most probable one
        self.assertEqual(one(0.9999), 0)                                     # u near 1 takes the least probable one


def tiny_config(bidirectional: bool) -> dict:
    text = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "linear_num_value_heads": 2,
            "linear_num_key_heads": 2, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "vocab_size": 300, "max_position_embeddings": 4096,
            "tie_word_embeddings": True, "eos_token_id": 2, "full_attention_interval": 4,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2], "rope_type": "default",
                                "rope_theta": 10000.0, "partial_rotary_factor": 0.25}}
    vision = {"model_type": "qwen3_5", "depth": 2, "hidden_size": 32, "intermediate_size": 64, "out_hidden_size": 64,
              "num_heads": 2, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
              "num_position_embeddings": 16, "deepstack_visual_indexes": [], "in_channels": 3}
    cfg = {"model_type": "qwen3_5", "text_config": text, "vision_config": vision, "image_token_id": IMAGE_ID,
           "video_token_id": 251, "vision_start_token_id": 252, "vision_end_token_id": 253, "tie_word_embeddings": True}
    if bidirectional:
        cfg["jev_prompt_attention"] = "bidirectional"
    return cfg


class FakeTokenizer:
    """Characters to ids 10..249, the image placeholder tokens to 250 (pad), 252 (start) and 253 (end); decode maps an id
    to a distinct character."""
    eos_token_id = 2
    SPECIAL = {IMAGE_PAD: IMAGE_ID, "<|vision_start|>": 252, "<|vision_end|>": 253}

    def encode(self, text, add_special_tokens=False):
        out = []
        for piece in re.split("(" + "|".join(re.escape(k) for k in self.SPECIAL) + ")", text):
            out += [self.SPECIAL[piece]] if piece in self.SPECIAL else [10 + ord(c) % 240 for c in piece]
        return out

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x4E00 + t) for t in ids)


def tiny_folder(bidirectional: bool, seed: int = 3) -> str:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    cfg = tiny_config(bidirectional)
    mx.random.seed(seed)
    model = Model(ModelConfig.from_dict(cfg))
    model.load_weights([(k, (mx.random.normal(v.shape) * 0.15) if v.ndim > 1 else v.astype(mx.float32))
                        for k, v in tree_flatten(model.parameters())], strict=False)
    mx.eval(model.parameters())
    path = tempfile.mkdtemp(prefix="startlux_tiny_mlx_")
    model.save_weights(os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return path


def photo(seed: int = 0, size=(64, 96)) -> str:
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((size[0], size[1], 3)) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@unittest.skipUnless(HAVE_MLX, "mlx-vlm is not installed")
class TinyModelTest(unittest.TestCase):
    """A random 4-layer model (3 linear-attention layers and 1 full-attention layer) with a 2-layer vision tower."""

    @classmethod
    def setUpClass(cls):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        cls.folders = {b: tiny_folder(b) for b in (False, True)}
        cls.backends = {}
        for b, path in cls.folders.items():
            be = M.MLXBackend(path, FakeTokenizer(), b, prefill_chunk=16)
            run(be.start())
            be.processor = SimpleNamespace(image_processor=Qwen3VLImageProcessor(patch_size=16))
            be.reads_images = True
            cls.backends[b] = be
        cls.media = load_media([photo(1), photo(2, (96, 64))], min_pixels=4096, max_pixels=16384)

    @classmethod
    def tearDownClass(cls):
        for be in cls.backends.values():
            run(be.close())

    def stock_hidden(self, be, ids, media=None):
        """The final hidden state of the last token in mlx-vlm's own pipeline (its input embeddings and positions, its
        decoder), float64. Matrix products here are done in float64: the GPU's float32 matmul of a whole sequence rounds
        coarser (about 1e-3) than the vector products the engine uses for the read-out."""
        import mlx.core as mx
        x = mx.array(ids)[None]
        if media is None:
            feats = be.model.get_input_embeddings(x)
        else:
            out = be.processor.image_processor(images=list(media.images), min_pixels=media.min_pixels,
                                               max_pixels=media.max_pixels)
            feats = be.model.get_input_embeddings(x, mx.array(out["pixel_values"]),
                                                  image_grid_thw=mx.array(out["image_grid_thw"]))
        h = be.body(x, inputs_embeds=feats.inputs_embeds, position_ids=feats.position_ids)
        return np.array(h[0, -1], dtype=np.float64)

    def head(self, be):
        import mlx.core as mx
        return np.array(be.body.embed_tokens.weight.astype(mx.float32), dtype=np.float64)       # tied embeddings

    def expanded_ids(self, be, prompt, media):
        """The prompt's token ids with every image placeholder expanded to the image's tokens."""
        if media is None:
            return be.tokenizer.encode(prompt)
        grids = be.processor.image_processor(images=list(media.images), min_pixels=media.min_pixels,
                                             max_pixels=media.max_pixels)["image_grid_thw"]
        parts = prompt.split(IMAGE_PAD)
        ids = be.tokenizer.encode(parts[0])
        for g, rest in zip(grids, parts[1:]):
            ids += [IMAGE_ID] * (int(np.prod(g)) // 4) + be.tokenizer.encode(rest)
        return ids

    def prompt(self, with_images):
        return "hello world " + (f"<|vision_start|>{IMAGE_PAD}<|vision_end|> " * 2 if with_images else "") + "what next?"

    def test_prompt_attention_is_in_effect(self):
        self.assertEqual(self.backends[False].check["first_position_change"], 0.0)
        self.assertGreater(self.backends[True].check["first_position_change"], 0.0)

    def test_causal_readout_matches_stock(self):
        be = self.backends[False]
        letters = [40, 41, 42]
        for media in (None, self.media):
            prompt = self.prompt(media is not None)
            got = np.array(run(be.readout(prompt, letters, media=media)))
            want = self.head(be)[letters] @ self.stock_hidden(be, self.expanded_ids(be, prompt, media), media)
            # the backend reads these prompts in chunks of 16 tokens (setUpClass); the GPU's float32 matmul rounds by about
            # 1e-4..1e-3 differently for different shapes (on the CPU the two agree to 1e-6)
            np.testing.assert_allclose(got, want, atol=2e-3)

    def test_bidirectional_readout_differs_from_causal_only_for_longer_prompts(self):
        a, b = self.backends[False], self.backends[True]
        self.assertTrue(np.allclose(run(a.readout("x", [40, 41])), run(b.readout("x", [40, 41])), atol=1e-4))
        prompt = self.prompt(False)
        self.assertFalse(np.allclose(run(a.readout(prompt, [40, 41])), run(b.readout(prompt, [40, 41])), atol=1e-4))
        withimg = self.prompt(True)
        self.assertFalse(np.allclose(run(a.readout(withimg, [40, 41], media=self.media)),
                                     run(b.readout(withimg, [40, 41], media=self.media)), atol=1e-4))

    def test_generation_follows_the_full_forward_pass(self):
        """Every token the engine samples (greedy, cache and decode positions) is a top token of the stock forward pass
        over the prompt and the tokens before it (ties within 5e-3 count: the two paths round differently)."""
        be = self.backends[False]
        greedy = Sampling(temperature=0.0, top_p=1.0, top_k=0)
        w = self.head(be)
        for media in (None, self.media):
            prompt = self.prompt(media is not None)
            out = run(be.generate(prompt, 12, greedy, [], 1, media=media))
            tokens = [ord(c) - 0x4E00 for c in out.text]
            self.assertEqual(out.tokens, len(tokens) + (out.finish_reason == "stop"))
            self.assertGreaterEqual(len(tokens), 6)
            ids = self.expanded_ids(be, prompt, media)
            for t, tok in enumerate(tokens):
                logits = w @ self.stock_hidden(be, ids + tokens[:t], media)
                self.assertGreaterEqual(logits[tok], logits.max() - 5e-3, f"token {t} of {len(tokens)}, media={media is not None}")

    def test_stop_string_budget_and_seed(self):
        be = self.backends[False]
        s = Sampling(temperature=1.0, top_p=0.95, top_k=20)
        a = run(be.generate("tell me", 20, s, [], 7))
        b = run(be.generate("tell me", 20, s, [], 7))
        c = run(be.generate("tell me", 20, s, [], 8))
        self.assertEqual((a.text, a.tokens), (b.text, b.tokens))
        self.assertNotEqual(a.text, c.text)
        if a.finish_reason == "length":
            self.assertEqual(a.tokens, 20)
        cut = a.text[3:5]
        d = run(be.generate("tell me", 20, s, [cut], 7))
        self.assertEqual(d.finish_reason, "stop")
        self.assertEqual(d.text, a.text[:a.text.index(cut)])

    def test_instant_does_not_generate_and_text_only_folders_refuse_images(self):
        with self.assertRaises(NotImplementedError):
            run(self.backends[True].generate("x", 4, Sampling(1.0, 1.0, 0), [], 1))
        be = self.backends[False]
        be.reads_images = False
        try:
            from startlux_decide.backends.base import BackendRequestError
            with self.assertRaises(BackendRequestError):
                run(be.readout(self.prompt(True), [40, 41], media=self.media))
        finally:
            be.reads_images = True


@unittest.skipUnless(HAVE_MLX and os.environ.get("STARTLUX_MLX_FOLDER"), "set STARTLUX_MLX_FOLDER to an MLX model folder")
class FolderTest(unittest.TestCase):
    """A real folder: it loads, the prompt attention of its flag is in effect, a read-out is a distribution."""

    def test_loads_and_reads(self):
        from startlux_decide.modelcfg import load_model_folder, load_tokenizer
        path = os.environ["STARTLUX_MLX_FOLDER"]
        folder = load_model_folder(path)
        be = M.MLXBackend(path, load_tokenizer(path), folder.bidirectional)
        run(be.start())
        try:
            self.assertEqual(be.describe()["prompt_attention"], "bidirectional" if folder.bidirectional else "causal")
            prompt = "Which is a fruit?\nA) apple\nB) stone\nAnswer:"
            z = np.array(run(be.readout(prompt, [32, 33])))
            self.assertEqual(z.shape, (2,))
            self.assertTrue(np.isfinite(z).all())
        finally:
            run(be.close())


if __name__ == "__main__":
    unittest.main()
