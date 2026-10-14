"""llama-server with images (backends/llamacpp.py): what /props says, the prompt object (text with the server's media
marker, each image a base64 PNG at the processor size), the token-count check, Instant folders (a non-causal
server context is needed), servers without an mmproj, the whole pipeline over the fake server, and that text-only
calls are exactly what they were. Needs aiohttp and Pillow."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import random
import unittest
import unittest.mock
import warnings

from startlux_decide.backends.base import BackendError, BackendRequestError, Sampling
from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.controls import Controls, RequestError
from startlux_decide.images import IMAGE_PLACEHOLDER, load_media, place_images, smart_resize
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import from_systemone, messages, plain_prompt

from .test_http_engines import FakeServer, aiohttp, llama_handler
from .test_images import INSTANT, VISION, Image, png
from .test_pipeline import QUESTIONS

MARKER = "<__media_Test1234__>"
LETTERS = list(range(65, 70))

if aiohttp is not None:
    from startlux_decide.backends import llamacpp
    from startlux_decide.backends.llamacpp import LlamaCppBackend


def vision_handler(bodies, tok, vision=True, off_by=0, other_tokenizer=0, tokenize_calls=None, whole=True):
    """A fake llama-server with a multimodal projector. `off_by`: it evaluates that many more prompt tokens than
    expected (an image resized); `other_tokenizer`: its tokenizer finds that many more tokens in every text piece;
    `whole`: it has patch 0003 (a prompt with images goes whole into one batch on a non-causal context)."""
    plain = llama_handler(True, bodies)

    def count(prompt):
        if not isinstance(prompt, dict):
            return len(prompt)
        n = sum(len(tok.encode(p)) + other_tokenizer for p in prompt["prompt_string"].split(MARKER))
        for d in prompt["multimodal_data"]:
            w, h = Image.open(io.BytesIO(base64.b64decode(d))).size
            n += 2 + (h // 32) * (w // 32)
        return n + off_by

    async def handle(url, body):
        if body is None:
            props = {"default_generation_settings": {"n_ctx": 32768}}
            if vision:
                props.update(modalities={"vision": True, "video": True, "audio": False}, media_marker=MARKER)
                if whole:
                    props["mtmd_prompt_whole_when_non_causal"] = True
            return 200, props
        if url.endswith("/tokenize"):
            tokenize_calls.append(body)
            return 200, {"tokens": tok.encode(body["content"]) + [0] * other_tokenizer}
        status, out = await plain(url, body)
        return status, dict(out, tokens_evaluated=count(body["prompt"]))
    return handle


@unittest.skipIf(aiohttp is None or Image is None, "needs aiohttp and Pillow")
class LlamaCppImagesTest(unittest.IsolatedAsyncioTestCase):
    def prompt(self, tok, n=1, state="look <image> now"):
        return plain_prompt(tok, messages(from_systemone(place_images(state, n), QUESTIONS["urgent"])))

    async def test_props_and_text_calls(self):
        tok = FakeTokenizer()
        sent = {}
        for vision in (False, True):
            bodies = []
            async with FakeServer(vision_handler(bodies, tok, vision, tokenize_calls=[])) as srv:
                backend = LlamaCppBackend(srv.url, tok)
                await backend.start()
                self.assertEqual(backend.reads_images, vision)
                self.assertEqual(backend.describe()["images"], vision)
                await backend.readout("Pick one.", LETTERS)
                await backend.generate("Think.", 50, Sampling(0.6, 0.95, 20), ["</think>"], 3)
                sent[vision] = bodies[:]
                await backend.close()
        self.assertEqual(sent[False], sent[True])                         # text calls: the same bodies
        self.assertEqual(sent[True][0]["prompt"], tok.encode("Pick one."))
        self.assertEqual(sent[True][1]["prompt"], tok.encode("Think."))
        self.assertNotIn("multimodal_data", json.dumps(sent[True]))

    async def test_readout_with_images(self):
        tok, bodies, calls = FakeTokenizer(), [], []
        async with FakeServer(vision_handler(bodies, tok, tokenize_calls=calls)) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            self.assertTrue(backend.reads_images)
            media = load_media([png(70, 50), png(1500, 1200, color=(0, 90, 0), uri=False)])
            prompt = self.prompt(tok, 2, "first <image> then <image>")
            z = await backend.readout(prompt, LETTERS, media=media)
            self.assertAlmostEqual(sum(math.exp(v) for v in z), 1.0, places=6)
            body = bodies[-1]
            want = prompt.replace(IMAGE_PLACEHOLDER, MARKER)
            self.assertEqual(body["prompt"]["prompt_string"], want)
            sizes = [Image.open(io.BytesIO(base64.b64decode(d))).size for d in body["prompt"]["multimodal_data"]]
            self.assertEqual(sizes, [(w, h) for h, w in media.sizes])             # at the processor size
            self.assertEqual(media.sizes[0], smart_resize(50, 70))                 # 70 x 50 grown to 65,536 pixels
            self.assertGreaterEqual(media.sizes[0][0] * media.sizes[0][1], 65536)
            self.assertLessEqual(media.sizes[1][0] * media.sizes[1][1], 1 << 20)
            self.assertEqual((body["n_predict"], body["n_probs"], body["top_k"], body["cache_prompt"],
                              body["post_sampling_probs"]), (1, 5, 5, False, True))
            self.assertEqual(body["logit_bias"], [[t, 100.0] for t in LETTERS])
            self.assertEqual(calls, [])                                           # the counts agreed: no /tokenize
            img0 = Image.open(io.BytesIO(base64.b64decode(body["prompt"]["multimodal_data"][0])))
            self.assertEqual(img0.mode, "RGB")
            # an image at the processor size goes out unchanged (lossless PNG)
            exact = Image.new("RGB", (320, 256), (10, 200, 30))
            exact.putpixel((3, 4), (255, 0, 1))
            buf = io.BytesIO()
            exact.save(buf, "PNG")
            media = load_media([base64.b64encode(buf.getvalue()).decode()])
            await backend.readout(self.prompt(tok), LETTERS, media=media)
            back = Image.open(io.BytesIO(base64.b64decode(bodies[-1]["prompt"]["multimodal_data"][0])))
            self.assertEqual(list(back.getdata()), list(exact.getdata()))
            # thoughts over images: the prompt grows, the object stays
            g = await backend.generate(self.prompt(tok) + "<think>\nhmm", 40, Sampling(0.6, 0.95, 20), ["</think>"],
                                       5, media=media)
            self.assertEqual(g.tokens, 7)
            self.assertEqual(bodies[-1]["prompt"]["prompt_string"],
                             (self.prompt(tok) + "<think>\nhmm").replace(IMAGE_PLACEHOLDER, MARKER))
            await backend.close()

    async def test_encoded_once_per_request(self):
        tok, bodies = FakeTokenizer(), []
        async with FakeServer(vision_handler(bodies, tok, tokenize_calls=[])) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            media = load_media([png(300, 200)])
            with unittest.mock.patch.object(llamacpp, "png_data", wraps=llamacpp.png_data) as spy:
                await asyncio.gather(*(backend.readout(self.prompt(tok, state=f"{k} <image>"), LETTERS, media=media)
                                       for k in range(4)))
                await backend.readout(self.prompt(tok), LETTERS, media=media)
            self.assertEqual(spy.call_count, 1)
            data = {json.dumps(b["prompt"]["multimodal_data"]) for b in bodies}
            self.assertEqual(len(data), 1)
            await backend.close()

    async def test_token_count_check(self):
        tok = FakeTokenizer()
        media = load_media([png(300, 200)])
        for kwargs, error in (({}, None), ({"off_by": 24}, "evaluated .* prompt tokens"),
                              ({"other_tokenizer": 1}, None)):
            calls = []
            async with FakeServer(vision_handler([], tok, tokenize_calls=calls, **kwargs)) as srv:
                backend = LlamaCppBackend(srv.url, tok)
                await backend.start()
                if error is None:
                    await backend.readout(self.prompt(tok), LETTERS, media=media)
                    await backend.generate(self.prompt(tok), 10, Sampling(0.6, 0.95, 20), [], 1, media=media)
                    self.assertEqual(len(calls) > 0, bool(kwargs))             # /tokenize only to explain a difference
                else:
                    with self.assertRaisesRegex(BackendError, error) as cm:
                        await backend.readout(self.prompt(tok), LETTERS, media=media)
                    self.assertIn("--image-max-tokens 1024", str(cm.exception))
                    with self.assertRaisesRegex(BackendError, error):
                        await backend.generate(self.prompt(tok), 10, Sampling(0.6, 0.95, 20), [], 1, media=media)
                await backend.close()

    async def test_refusals(self):
        tok = FakeTokenizer()
        media = load_media([png(64, 64)])
        async with FakeServer(vision_handler([], tok, vision=False, tokenize_calls=[])) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            with self.assertRaisesRegex(BackendRequestError, "--mmproj"):
                await backend.readout(self.prompt(tok), LETTERS, media=media)
            with self.assertRaisesRegex(RequestError, "reads text only"):         # the pipeline answers 422 first
                await Decider(VISION, backend).decide("<image>", {"q": QUESTIONS["urgent"]}, images=[png(64, 64)])
            await backend.close()
        async with FakeServer(vision_handler([], tok, tokenize_calls=[])) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            with self.assertRaisesRegex(BackendRequestError, "2 image places for 1 images"):
                await backend.readout(self.prompt(tok, 2, "<image> <image>"), LETTERS, media=media)
            with self.assertRaisesRegex(BackendRequestError, "media marker"):
                await backend.readout(self.prompt(tok, state=f"{MARKER} <image>"), LETTERS, media=media)
            await backend.close()

    async def test_instant_folders(self):
        """Instant + images: only a non-causal server context reads the image with the text in one micro-batch."""
        tok, media = FakeTokenizer(), load_media([png(64, 64)])
        for mode, ok in (("request", False), ("server", True), ("off", True)):
            bodies = []
            with unittest.mock.patch.dict(os.environ, {"STARTLUX_LLAMACPP_BIDIR": mode}):
                async with FakeServer(vision_handler(bodies, tok, tokenize_calls=[])) as srv:
                    backend = LlamaCppBackend(srv.url, tok, bidirectional=True)
                    await backend.start()
                    if ok:
                        await backend.readout(self.prompt(tok), LETTERS, media=media)
                        self.assertTrue(isinstance(bodies[-1]["prompt"], dict) and
                                        "prompt_bidirectional" not in bodies[-1], mode)
                    else:
                        with self.assertRaisesRegex(BackendRequestError, "non-causal.*STARTLUX_LLAMACPP_BIDIR=server"):
                            await backend.readout(self.prompt(tok), LETTERS, media=media)
                        self.assertEqual(bodies, [])                              # nothing was sent
                    await backend.close()

    async def test_instant_images_with_auto_mode(self):
        """With the default (auto), a server that reports patch 0003 reads an Instant folder with images as `server`
        and nothing needs to be set; a server without the flag stays `request`, which refuses images and names `server`."""
        tok, media = FakeTokenizer(), load_media([png(64, 64)])
        for whole in (True, False):
            bodies = []
            with unittest.mock.patch.dict(os.environ):
                os.environ.pop("STARTLUX_LLAMACPP_BIDIR", None)
                async with FakeServer(vision_handler(bodies, tok, tokenize_calls=[], whole=whole)) as srv:
                    backend = LlamaCppBackend(srv.url, tok, bidirectional=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        await backend.start()
                    self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
                    self.assertEqual(backend.bidir_mode, "server" if whole else "request")
                    if whole:
                        await backend.readout(self.prompt(tok), LETTERS, media=media)
                        self.assertTrue(isinstance(bodies[-1]["prompt"], dict) and
                                        "prompt_bidirectional" not in bodies[-1])
                        self.assertEqual(backend.describe()["image_prompt"], "whole (one micro-batch)")
                    else:
                        with self.assertRaisesRegex(BackendRequestError, "non-causal.*STARTLUX_LLAMACPP_BIDIR=server"):
                            await backend.readout(self.prompt(tok), LETTERS, media=media)
                        self.assertEqual(bodies, [])                              # nothing was sent
                    await backend.close()

    async def test_split_image_prompts_are_flagged(self):
        """An Instant folder with images on a server without patch 0003 gets a warning and says so in describe()."""
        tok = FakeTokenizer()
        for whole in (True, False):
            with unittest.mock.patch.dict(os.environ, {"STARTLUX_LLAMACPP_BIDIR": "server"}):
                async with FakeServer(vision_handler([], tok, tokenize_calls=[], whole=whole)) as srv:
                    backend = LlamaCppBackend(srv.url, tok, bidirectional=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        await backend.start()
                    flagged = [w for w in caught if issubclass(w.category, RuntimeWarning)]
                    self.assertEqual(len(flagged), 0 if whole else 1)
                    self.assertEqual(backend.describe()["image_prompt"],
                                     "whole (one micro-batch)" if whole else "split (stock server)")
                    await backend.close()
            async with FakeServer(vision_handler([], tok, tokenize_calls=[], whole=False)) as srv:
                backend = LlamaCppBackend(srv.url, tok)                       # AutoThink (causal): nothing to flag
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    await backend.start()
                self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
                self.assertNotIn("image_prompt", backend.describe())
                await backend.close()

    async def test_pipeline_over_the_fake_server(self):
        """Decider -> LlamaCppBackend -> fake server: one-pass image answers for Instant, and a thought for AutoThink
        whose calls all carry the image."""
        tok = FakeTokenizer()
        img = png(320, 256)
        with unittest.mock.patch.dict(os.environ, {"STARTLUX_LLAMACPP_BIDIR": "server"}):
            bodies = []
            async with FakeServer(vision_handler(bodies, tok, tokenize_calls=[])) as srv:
                backend = LlamaCppBackend(srv.url, tok, bidirectional=True)
                await backend.start()
                d = Decider(INSTANT, backend)
                out = await d.decide("<image>", {"a": QUESTIONS["urgent"], "b": QUESTIONS["team"]},
                                     Controls(think_mode="never", details=True), images=[img])
                self.assertEqual(len(bodies), 2)
                self.assertTrue(all(isinstance(b["prompt"], dict) for b in bodies))
                self.assertEqual(set(out.answers), {"a", "b"})
                image_tokens = (256 // 32) * (320 // 32)
                self.assertGreaterEqual(out.usage["input_tokens"], 2 * image_tokens)    # both questions read the image
                await backend.close()
        bodies = []
        async with FakeServer(vision_handler(bodies, tok, tokenize_calls=[])) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            d = Decider(VISION, backend)
            out = await d.decide("Photo: <image>", {"q": QUESTIONS["urgent"]},
                                 Controls(think_mode="always", details=True, budget=64), images=[img])
            kinds = [b["n_predict"] for b in bodies]
            self.assertEqual(kinds[0], 1)
            self.assertEqual(kinds[-1], 1)
            self.assertGreater(len(kinds), 2)                                     # read-out, thought chunks, read-out
            self.assertTrue(all(isinstance(b["prompt"], dict) for b in bodies))
            self.assertTrue(out.answers["q"]["details"]["thought"]["used"])
            await backend.close()


if __name__ == "__main__":
    unittest.main()
