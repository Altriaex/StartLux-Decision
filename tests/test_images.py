"""Images in the evidence: their places in the state, the resize rule and token counts, decoding, the prompts, the
calls of the pipeline (every engine call of the request carries the images), the 422 of text-only engines and
folders, and the vLLM client's chat call (the prompt rebuilt from its parts is the package's prompt). Decoding needs
Pillow; the pass-through template check needs jinja2."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import unittest
from dataclasses import replace

from startlux_decide.backends.fake import FakeBackend, FakeTokenizer
from startlux_decide.backends.vllm import VLLMBackend
from startlux_decide.controls import Controls, RequestError
from startlux_decide.images import (IMAGE_MARK, IMAGE_PLACEHOLDER, PASS_THROUGH_TEMPLATE, SLOT, multimodal_messages,
                                    place_images, smart_resize, split_prompt)
from startlux_decide.launch import VLLMServers
from startlux_decide.modelcfg import load_model_folder
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import count_tokens, from_systemone, messages, plain_prompt, think_prompt

from .fake_vllm import FakeVLLM
from .folders import make_folder
from .test_pipeline import AUTO, QUESTIONS, closing

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import jinja2
except ImportError:
    jinja2 = None

VISION = replace(AUTO, vision=True)
INSTANT = replace(VISION, prompt_attention="bidirectional", auto_think=None)


def png(width: int, height: int, color=(200, 30, 30), uri: bool = True) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}" if uri else b64


def run(coro):
    return asyncio.run(coro)


class PlacesAndSizesTest(unittest.TestCase):
    def test_places(self):
        self.assertEqual(place_images(f"before {IMAGE_MARK} after", 1), f"before {SLOT} after")
        self.assertEqual(place_images(f"{IMAGE_MARK}{IMAGE_MARK}", 2), SLOT + SLOT)
        self.assertEqual(place_images("the parcel", 1), f"{SLOT}\n\nthe parcel")
        self.assertEqual(place_images("the parcel", 2), f"Image 1: {SLOT}\nImage 2: {SLOT}\n\nthe parcel")
        self.assertEqual(place_images("", 1), SLOT)
        self.assertEqual(place_images('{"photo": "<image>", "id": 7}', 1), '{"photo": "' + SLOT + '", "id": 7}')
        with self.assertRaisesRegex(ValueError, "2 <image> marks for 1 images"):
            place_images(f"{IMAGE_MARK} and {IMAGE_MARK}", 1)

    def test_resize_rule(self):
        self.assertEqual(smart_resize(1024, 1024), (1024, 1024))             # 1,024 tokens
        self.assertEqual(smart_resize(10, 10), (256, 256))                   # up to 65,536 pixels
        h, w = smart_resize(3000, 2000)
        self.assertTrue(h % 32 == 0 and w % 32 == 0 and h * w <= 1 << 20 and abs(h / w - 1.5) < 0.05)
        self.assertEqual(smart_resize(4000, 4000, max_pixels=1 << 22), (2048, 2048))
        with self.assertRaisesRegex(ValueError, "factor of 200"):
            smart_resize(10, 3000)

    def test_messages_and_prompts(self):
        tok = FakeTokenizer()
        row = from_systemone(place_images("the parcel", 1), QUESTIONS["urgent"])
        msgs = messages(row)
        parts = multimodal_messages(msgs)[1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image", "text"])
        for render in (plain_prompt, think_prompt):
            text = render(tok, msgs)
            self.assertEqual(text.count(IMAGE_PLACEHOLDER), 1)
            self.assertEqual(text.replace(IMAGE_PLACEHOLDER, "@"),
                             render(tok, messages(from_systemone("@\n\nthe parcel", QUESTIONS["urgent"]))))
            self.assertEqual(IMAGE_PLACEHOLDER.join(split_prompt(text)), text)

    @unittest.skipIf(jinja2 is None, "needs jinja2")
    def test_pass_through_template(self):
        """The template the vLLM client sends rebuilds the prompt from the parts of its chat body."""
        tok = FakeTokenizer()
        prompt = plain_prompt(tok, messages(from_systemone(place_images("a <image> b <image>", 2), QUESTIONS["team"])))
        backend = VLLMBackend(["8001"], tok, transport=FakeVLLM())
        backend.model_name = "m"

        class TwoImages:
            def __len__(self):
                return 2

            def data_uris(self):
                return ["data:image/png;base64,AA==", "data:image/png;base64,AB=="]
            processor_size = {"longest_edge": 1 << 20, "shortest_edge": 65536}
        body = backend._body(prompt, TwoImages(), max_tokens=1)
        rendered = jinja2.Environment().from_string(PASS_THROUGH_TEMPLATE).render(
            messages=[{"role": "user", "content": [p if p["type"] == "text" else {"type": "image"}
                                                    for p in body["messages"][0]["content"]]}])
        self.assertEqual(rendered, prompt)


@unittest.skipIf(Image is None, "needs Pillow")
class MediaTest(unittest.TestCase):
    def test_decode_and_count(self):
        from startlux_decide.images import load_media
        m = load_media([png(1024, 1024), png(64, 32, uri=False)])
        self.assertEqual((len(m), m.sizes, m.tokens), (2, [(1024, 1024), (192, 384)], [1024, 72]))   # 64 x 32 up
        self.assertTrue(m.data_uris()[1].startswith("data:image/png;base64,"))
        for bad, why in (("not base64 !", "base64"), (base64.b64encode(b"text").decode(), "not a readable image"),
                         (7, "base64 string")):
            with self.assertRaisesRegex(ValueError, why):
                load_media([bad])
        with self.assertRaisesRegex(ValueError, "at most 2 images"):
            load_media([png(40, 40)] * 3, max_images=2)
        with self.assertRaisesRegex(ValueError, "a list"):
            load_media(png(40, 40))


@unittest.skipIf(Image is None, "needs Pillow")
class PipelineImagesTest(unittest.TestCase):
    def test_every_call_carries_the_images(self):
        backend = FakeBackend(script=closing, reads_images=True, streams=True)
        d = Decider(VISION, backend)
        img = png(320, 256)
        out = run(d.decide("Photo taken at delivery: <image>", {"q": QUESTIONS["urgent"]},
                           Controls(think_mode="always", details=True), images=[img]))
        kinds = [c[0] for c in backend.calls]
        self.assertEqual(kinds, ["readout", "generate_stream", "readout"])
        self.assertTrue(all(c[-1] == 1 and IMAGE_PLACEHOLDER in c[1] for c in backend.calls))
        self.assertIn("Photo taken at delivery: " + IMAGE_PLACEHOLDER, backend.calls[0][1])
        image_tokens = (256 // 32) * (320 // 32)                                # the placeholder's pad token
        self.assertEqual(out.usage["input_tokens"], count_tokens(d.tok, backend.calls[0][1]) + image_tokens - 1)
        other = run(Decider(VISION, FakeBackend(script=closing, reads_images=True)).decide(
            "Photo taken at delivery: <image>", {"q": QUESTIONS["urgent"]}, Controls(think_mode="never"),
            images=[png(320, 256, color=(0, 0, 255))]))
        self.assertNotEqual(other.answers["q"]["noul"], out.answers["q"]["noul"])        # the image is read

    def test_knockout_and_instant(self):
        crit = {f"opt{i}": None for i in range(40)}
        backend = FakeBackend(reads_images=True)
        run(Decider(INSTANT, backend).decide("<image>", {"q": {"type": "choice", "instructions": "Which?",
                                                               "criteria": crit}}, images=[png(64, 64)]))
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(all(c[-1] == 1 for c in backend.calls))

    def test_refusals(self):
        img = png(64, 64)
        cases = [(Decider(VISION, FakeBackend()), "reads text only", ["body", "images"]),            # engine
                 (Decider(AUTO, FakeBackend(reads_images=True)), "no vision tower", ["body", "images"]),
                 (Decider(VISION, FakeBackend(reads_images=True), max_images=0), "reads text only", ["body", "images"])]
        for d, why, loc in cases:
            with self.assertRaises(RequestError) as cm:
                run(d.decide("x", {"q": QUESTIONS["urgent"]}, images=[img]))
            self.assertIn(why, str(cm.exception))
            self.assertEqual(cm.exception.loc, loc)
        d = Decider(VISION, FakeBackend(reads_images=True))
        with self.assertRaises(RequestError) as cm:
            run(d.decide("<image> <image>", {"q": QUESTIONS["urgent"]}, images=[img]))
        self.assertEqual(cm.exception.loc, ["body", "state"])
        with self.assertRaises(RequestError) as cm:
            run(d.decide("x", {"q": QUESTIONS["urgent"]}, images=["%%%"]))
        self.assertEqual(cm.exception.loc, ["body", "images"])
        run(d.decide("x", {"q": QUESTIONS["urgent"]}, images=[]))                   # no images: text only


@unittest.skipIf(Image is None, "needs Pillow")
class VLLMImagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_calls(self):
        tok = FakeTokenizer()
        fake = FakeVLLM(version="0.31.0", kinds=("close",))
        backend = VLLMBackend(["8001"], tok, transport=fake)
        await backend.start()
        self.assertTrue(backend.reads_images and backend.caches_prefixes)
        d = Decider(VISION, backend)
        imgs = [png(64, 64), png(96, 64, color=(0, 90, 0))]
        out = await d.decide("first <image> then <image>", {"q": QUESTIONS["urgent"]},
                             Controls(think_mode="always", details=True, thought_text=True), images=imgs)
        prompts = [p for p, _, _ in fake.chat]
        self.assertEqual(len(fake.chat), 3)                                  # read-out, thought stream, read-out
        self.assertTrue(all(len(u) == 2 for _, u, _ in fake.chat))
        self.assertTrue(fake.chat[1][2]["stream"])
        expected = plain_prompt(tok, messages(from_systemone(place_images("first <image> then <image>", 2),
                                                             QUESTIONS["urgent"])))
        self.assertEqual(prompts[0], expected)
        self.assertEqual(out.answers["q"]["details"]["thought"]["generation"], "stream")
        old = VLLMBackend(["8001"], tok, transport=FakeVLLM(version="0.19.1"))
        await old.start()
        self.assertFalse(old.reads_images)
        with self.assertRaisesRegex(RequestError, "reads text only"):
            await Decider(VISION, old).decide("x", {"q": QUESTIONS["urgent"]}, images=imgs)
        off = VLLMBackend(["8001"], tok, transport=FakeVLLM(version="0.31.0"), images=False)
        await off.start()
        self.assertFalse(off.reads_images)


@unittest.skipIf(Image is None, "needs Pillow")
class ServerImagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_post_with_images(self):
        try:
            from aiohttp.test_utils import TestClient, TestServer

            from startlux_decide.server import create_app
        except ImportError:
            self.skipTest("needs aiohttp")
        path = make_folder(config={"vision_config": {"patch_size": 16, "spatial_merge_size": 2}})
        backend = FakeBackend(script=closing, reads_images=True)
        client = TestClient(TestServer(create_app(load_model_folder(path), backend, "m", images={"max_images": 2})))
        await client.start_server()
        try:
            body = {"state": "Photo taken at delivery: <image>", "questions": {"q": QUESTIONS["urgent"]},
                    "images": [png(64, 64)], "think_mode": "never"}
            r = await client.post("/v1/systemone", json=body)
            self.assertEqual(r.status, 200, await r.text())
            self.assertEqual(backend.calls[-1][-1], 1)
            h = await (await client.get("/health")).json()
            self.assertTrue(h["images"])
            for bad, loc in (({"images": [png(64, 64)] * 3}, "images"), ({"images": "x"}, "images"),
                             ({"state": "<image> <image>"}, "state")):
                r = await client.post("/v1/systemone", json={**body, **bad})
                self.assertEqual(r.status, 422)
                self.assertEqual((await r.json())["detail"][0]["loc"], ["body", loc])
        finally:
            await client.close()
            shutil.rmtree(path, ignore_errors=True)


class LaunchImagesTest(unittest.TestCase):
    def test_flags(self):
        vision = {"vision_config": {"patch_size": 16, "spatial_merge_size": 2},
                  "text_config": {"max_position_embeddings": 262144}}
        folders = [make_folder(config=vision), make_folder("instant", config=vision), make_folder()]
        try:
            auto, inst, text = folders
            self.assertEqual((load_model_folder(auto).vision, load_model_folder(auto).native_context),
                             (True, 262144))
            cmd = VLLMServers(auto, ["0"], python="py", vllm_version="0.31.0").plan()[0][0]
            value = lambda c, f: c[c.index(f) + 1]                       # noqa: E731
            self.assertEqual(value(cmd, "--limit-mm-per-prompt"), '{"image":8,"video":0}')
            self.assertEqual(json.loads(value(cmd, "--mm-processor-kwargs")),
                             {"size": {"longest_edge": 1048576, "shortest_edge": 65536}})
            self.assertIn("--trust-request-chat-template", cmd)
            self.assertEqual(value(cmd, "--max-model-len"), "262144")
            self.assertEqual(value(cmd, "--max-num-batched-tokens"), "8192")
            cmd = VLLMServers(inst, ["0"], python="py", vllm_version="0.31.0").plan()[0][0]
            self.assertEqual((value(cmd, "--max-model-len"), value(cmd, "--max-num-batched-tokens")),
                             ("131072", "131072"))                           # Instant: a whole prompt in one step
            cmd = VLLMServers(inst, ["0"], python="py", vllm_version="0.31.0", max_model_len=262144).plan()[0][0]
            self.assertEqual(value(cmd, "--max-num-batched-tokens"), "262144")
            self.assertIn("--no-enable-chunked-prefill", cmd)
            self.assertEqual(value(cmd, "--limit-mm-per-prompt"), '{"image":8,"video":0}')
            self.assertIn("--skip-mm-profiling", cmd)                   # Instant with images
            self.assertNotIn("--skip-mm-profiling", VLLMServers(auto, ["0"], python="py",
                                                                vllm_version="0.31.0").plan()[0][0])
            for s in (VLLMServers(auto, ["0"], python="py", vllm_version="0.19.1"),
                      VLLMServers(auto, ["0"], python="py", vllm_version="0.31.0", max_images=0),
                      VLLMServers(text, ["0"], python="py", vllm_version="0.31.0")):
                cmd = s.plan()[0][0]
                self.assertEqual(value(cmd, "--limit-mm-per-prompt"), '{"image":0,"video":0}')
                self.assertNotIn("--trust-request-chat-template", cmd)
            s = VLLMServers(auto, ["0"], python="py", vllm_version="0.31.0", max_model_len=65536)
            self.assertEqual(s.plan()[0][0][s.plan()[0][0].index("--max-model-len") + 1], "65536")
            self.assertEqual(VLLMServers(text, ["0"], python="py").max_model_len, 65536)
            self.assertEqual(VLLMServers(auto, ["0"], python="py", vllm_version="0.19.1").max_model_len, 65536)
        finally:
            for f in folders:
                shutil.rmtree(f, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
