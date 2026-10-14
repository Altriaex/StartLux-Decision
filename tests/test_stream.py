"""Thoughts as one streamed request (thought_generation "stream"): on the fake engine (closed thought, a loop cut at
1,024 tokens, the budget cut, the stream closed early, the loop check where the chunked path makes it), the choice of
the mode, and the vLLM client on a fake vLLM at the JSON level: version probe, request body, and the same answers as
the chunked path on random requests."""
from __future__ import annotations

import asyncio
import json
import random
import string
import unittest
from dataclasses import replace
from unittest import mock

from startlux_decide import pipeline
from startlux_decide.backends.base import Sampling
from startlux_decide.backends.fake import FakeBackend, FakeTokenizer
from startlux_decide.backends.vllm import VLLMBackend, parse_version
from startlux_decide.controls import Controls, RequestError, parse_controls
from startlux_decide.modelcfg import AutoThink
from startlux_decide.pipeline import CLOSE, CUT, Decider

from .fake_vllm import KINDS, FakeVLLM
from .test_golden import MODES, folder_for, random_question, random_settings, random_state
from .test_pipeline import AUTO, QUESTIONS, closing

UNIT = "The answer is B.\n"


def prose(n: int) -> list[str]:
    """n tokens of random letter words with a line break every 9 words: no loop rule fires on them."""
    r = random.Random(5)
    return ["".join(r.choice(string.ascii_lowercase) for _ in range(6)) + ("\n" if i % 9 == 8 else " ")
            for i in range(n)]


def loop_after(n: int):
    """A thought of n prose tokens, then the same short line forever (the script ignores prompt and seed)."""
    def script(prompt, seed, max_tokens):
        return prose(n) + [UNIT] * max_tokens, False
    return script


def plain_prose(prompt, seed, max_tokens):
    return prose(max_tokens + 10), False


def run(coro):
    return asyncio.run(coro)


def answer(a: dict) -> dict:
    """An answer without its details (timings differ between runs)."""
    return {k: v for k, v in a.items() if k != "details"}


def decide(backend, folder=AUTO, controls=None, questions=None):
    d = Decider(folder, backend)
    ctl = controls or Controls(think_mode="always", details=True, thought_text=True)
    return run(d.decide("charged twice", questions or {"q": QUESTIONS["urgent"]}, ctl))


class FakeEngineStreamTest(unittest.TestCase):
    def test_closed_thought(self):
        backend = FakeBackend(script=closing, streams=True)
        out = decide(backend)
        th = out.answers["q"]["details"]["thought"]
        self.assertEqual((th["generation"], th["ended"], th["tokens"]), ("stream", "closed", 5))
        self.assertEqual(th["text"],
                         "Let me work through this carefully, step by step.\nBilling handles double charges.")
        self.assertEqual([c[0] for c in backend.calls], ["readout", "generate_stream", "readout"])
        self.assertTrue(backend.calls[-1][1].endswith("Billing handles double charges." + CLOSE))
        self.assertEqual(backend.aborted, 0)
        chunked = decide(FakeBackend(script=closing))                   # the same thought in chunks
        self.assertEqual(chunked.answers["q"]["details"]["thought"]["generation"], "chunked")
        self.assertEqual(answer(chunked.answers["q"]), answer(out.answers["q"]))

    def test_request_is_the_first_chunk(self):
        stream, chunks = FakeBackend(script=plain_prose, streams=True), FakeBackend(script=plain_prose)
        decide(stream)
        decide(chunks)
        s = [c for c in stream.calls if c[0] == "generate_stream"]
        g = [c for c in chunks.calls if c[0] == "generate"]
        self.assertEqual(len(s), 1)
        self.assertEqual(len(g), 4)                                     # 4,096 tokens in chunks of 1,024
        self.assertEqual(s[0][1], g[0][1])                              # prompt with the forced prefix
        self.assertEqual(s[0][3:], g[0][3:])                            # sampling, stop string, seed
        self.assertEqual((s[0][2], g[0][2]), (4096, 1024))              # one request for the whole budget

    def test_loop_cut_at_1024(self):
        chunked = decide(FakeBackend(script=loop_after(600)))
        cth = chunked.answers["q"]["details"]["thought"]
        self.assertEqual((cth["ended"], cth["tokens"]), ("loop", 1024))
        self.assertTrue(cth["text"].endswith(UNIT * 3 + UNIT.strip()[:8]) or cth["text"].count(UNIT.strip()) >= 4)
        for burst in (1, 3, 4):
            with self.subTest(burst=burst):
                backend = FakeBackend(script=loop_after(600), streams=True, burst=burst)
                out = decide(backend)
                th = out.answers["q"]["details"]["thought"]
                first = -(-1024 // burst) * burst                       # the first event at or past 1,024 tokens
                self.assertEqual((th["ended"], th["tokens"]), ("loop", first))
                self.assertEqual(backend.aborted, 1)                    # the stream was closed at the cut
                if burst == 1:                                          # the same text at the check: the same cut
                    self.assertEqual(th["text"], cth["text"])
                    self.assertEqual(answer(out.answers["q"]), answer(chunked.answers["q"]))
                else:   # checked a few tokens later: cut by the same rule, inside the loop, within a few characters
                    short, long_ = sorted((th["text"], cth["text"]), key=len)
                    self.assertTrue(long_.startswith(short))
                    self.assertLessEqual(len(long_) - len(short), 2 * len(UNIT))

    def test_budget_cut_and_check_points(self):
        folder = replace(AUTO, auto_think=AutoThink(budget=300, chunk_tokens=100))
        seen = []

        def looping(text):
            seen.append(text.count(" ") + text.count("\n"))             # one space or line break per prose token
            return False
        backend = FakeBackend(script=plain_prose, streams=True)
        with mock.patch.object(pipeline, "looping", looping):
            out = decide(backend, folder, Controls(think_mode="always", budget=300, details=True))
        th = out.answers["q"]["details"]["thought"]
        self.assertEqual((th["generation"], th["ended"], th["tokens"]), ("stream", "budget", 300))
        self.assertEqual(seen, [100, 200, 300])                         # as the chunked path: after every 100 tokens
        self.assertEqual([c[2] for c in backend.calls if c[0] == "generate_stream"], [300])
        self.assertTrue(backend.calls[-1][1].endswith(CUT))
        self.assertEqual(backend.aborted, 0)

    def test_loop_found_when_the_budget_ends(self):
        """No loop at 100 and 200 tokens, one at the budget (300): ended as a loop, as in the chunked path."""
        folder = replace(AUTO, auto_think=AutoThink(budget=300, chunk_tokens=100))
        ctl = Controls(think_mode="always", budget=300, details=True, thought_text=True)
        backend = FakeBackend(script=loop_after(230), streams=True)
        th = decide(backend, folder, ctl).answers["q"]["details"]["thought"]
        self.assertEqual((th["ended"], th["tokens"]), ("loop", 300))
        self.assertEqual(backend.aborted, 0)                            # the stream had ended by itself
        self.assertTrue(backend.calls[-1][1].endswith(CLOSE))

    def test_stream_closed_early_stops_reading(self):
        """The consumer stops at the cut: the engine produced no event after it."""
        produced = []

        class Counting(FakeBackend):
            async def generate_stream(self, *a):
                async for ev in super().generate_stream(*a):
                    produced.append(ev.tokens)
                    yield ev
        backend = Counting(script=loop_after(600), streams=True)
        th = decide(backend).answers["q"]["details"]["thought"]
        self.assertEqual(th["tokens"], 1024)
        self.assertEqual(produced[-1], 1024)
        self.assertEqual(backend.aborted, 1)

    def test_no_loop_check_without_chunk_tokens(self):
        folder = replace(AUTO, auto_think=AutoThink(budget=2000, chunk_tokens=0))
        backend = FakeBackend(script=loop_after(10), streams=True)
        th = decide(backend, folder, Controls(think_mode="always", budget=2000, details=True)).answers["q"]
        self.assertEqual((th["details"]["thought"]["ended"], th["details"]["thought"]["tokens"]), ("budget", 2000))


class ModeChoiceTest(unittest.TestCase):
    def test_auto_follows_the_engine(self):
        for streams, mode in ((True, "stream"), (False, "chunked")):
            backend = FakeBackend(script=closing, streams=streams)
            d = Decider(AUTO, backend)
            self.assertEqual(d.generation(d.defaults), mode)
            th = run(d.decide("x", {"q": QUESTIONS["urgent"]}, Controls(think_mode="always", details=True)))
            self.assertEqual(th.answers["q"]["details"]["thought"]["generation"], mode)

    def test_request_field(self):
        backend = FakeBackend(script=closing, streams=True)
        d = Decider(AUTO, backend)
        ctl = d.controls({"thought_generation": "chunked", "think_mode": "always", "return_details": True})
        out = run(d.decide("x", {"q": QUESTIONS["urgent"]}, ctl))
        self.assertEqual(out.answers["q"]["details"]["thought"]["generation"], "chunked")
        self.assertNotIn("generate_stream", [c[0] for c in backend.calls])
        old = Decider(AUTO, FakeBackend(script=closing))
        with self.assertRaisesRegex(RequestError, "does not stream"):
            old.controls({"thought_generation": "stream"})
        with self.assertRaises(RequestError):
            run(old.decide("x", {"q": QUESTIONS["urgent"]}, Controls(think_mode="always", thought_generation="stream")))
        run(old.decide("x", {"q": QUESTIONS["urgent"]}, Controls(think_mode="never", thought_generation="stream")))
        for bad in ("streamed", 1, True):
            with self.assertRaises(RequestError):
                d.controls({"thought_generation": bad})
        self.assertEqual(parse_controls({}, Controls(), 4096, True).thought_generation, "auto")
        with self.assertRaisesRegex(ValueError, "does not stream"):
            Decider(AUTO, FakeBackend(), Controls(thought_generation="stream"))


class VLLMClientStreamTest(unittest.IsolatedAsyncioTestCase):
    async def backend(self, fake, **kw):
        b = VLLMBackend(["http://127.0.0.1:8001"], FakeTokenizer(), transport=fake, **kw)
        await b.start()
        return b

    async def test_version_decides_streams_and_waves(self):
        cases = [("0.31.0", True, False), ("0.32.1rc2", True, False), ("0.19.1", False, True), (None, False, True)]
        for version, streams, waves in cases:
            b = await self.backend(FakeVLLM(version=version))
            self.assertEqual((b.streams, b.waves), (streams, waves), version)
            self.assertEqual(b.describe()["vllm_versions"], [version])
            await b.close()
        b = await self.backend(FakeVLLM(version="0.31.0"), waves=True)
        self.assertEqual((b.streams, b.waves), (True, True))
        await b.close()
        self.assertEqual((parse_version("0.31.0"), parse_version("v0.19.1+cu129"), parse_version("dev")),
                         ((0, 31, 0), (0, 19, 1), None))

    async def test_stream_body_and_events(self):
        fake = FakeVLLM(version="0.31.0", kinds=("close",), burst=4)
        b = await self.backend(fake)
        prompt = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nLet me think.\n"
        events = [ev async for ev in b.generate_stream(prompt, 4096, Sampling(0.6, 0.95, 20), ("</think>",), 7)]
        body = json.loads(fake.log[-1])
        self.assertEqual(body, {"model": "m", "prompt": prompt, "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95,
                                "top_k": 20, "stop": ["</think>"], "seed": 7, "stream": True,
                                "stream_options": {"include_usage": True, "continuous_usage_stats": True}})
        self.assertTrue(all(e.finish_reason is None for e in events[:-1]))
        self.assertEqual(events[-1].finish_reason, "stop")
        full = fake.generate(dict(body, stream=False))                  # the same generation in one answer
        self.assertEqual("".join(e.text for e in events), full["choices"][0]["text"])
        self.assertEqual(events[-1].tokens, full["usage"]["completion_tokens"])
        await b.close()

    async def test_closing_the_stream_ends_it(self):
        fake = FakeVLLM(version="0.31.0", kinds=("loop_short",))
        b = await self.backend(fake)
        prompt = "<|im_start|>assistant\n<think>\nP\n"
        stream = b.generate_stream(prompt, 4096, Sampling(0.6, 0.95, 20), ("</think>",), 1)
        async for ev in stream:
            if ev.tokens >= 10:
                break
        await stream.aclose()
        self.assertEqual(fake.aborted, 1)
        await b.close()

    async def test_same_answers_as_chunks(self):
        """Random requests and settings: streamed thoughts (one event per token) give the same thoughts, token counts
        and probabilities as 1,024-token chunks, because the loop detector runs at the same token counts."""
        r = random.Random(1011)
        stats = {"stream": 0, "loop": 0, "closed": 0, "budget": 0}
        for case in range(400):
            s = random_settings(r)
            state = random_state(r)
            questions = {f"q{i}": random_question(r) for i in range(r.choice([1, 1, 2, 3]))}
            tok = FakeTokenizer(open_think=s.open_think)
            got = {}
            for version in ("0.31.0", "0.19.1"):
                fake = FakeVLLM(s.max_len, drop_rate=s.drop, version=version)
                b = VLLMBackend(["8001"], tok, waves=False, transport=fake)
                await b.start()
                controls = Controls(think_mode=MODES[s.mode], gate=s.tau, budget=s.budget, mix=s.mix, details=True,
                                    thought_text=True)
                got[version] = await Decider(folder_for(s), b).decide(state, questions, controls)
                await b.close()
            a, c = got["0.31.0"], got["0.19.1"]
            with self.subTest(case=case):
                self.assertEqual(a.usage, c.usage)
                for key in questions:
                    x, y = a.answers[key], c.answers[key]
                    tx, ty = x["details"]["thought"], y["details"]["thought"]
                    self.assertEqual({k: v for k, v in x.items() if k != "details"},
                                     {k: v for k, v in y.items() if k != "details"})
                    self.assertEqual({k: tx.get(k) for k in ("used", "tokens", "ended", "text")},
                                     {k: ty.get(k) for k in ("used", "tokens", "ended", "text")})
                    if tx["used"]:
                        self.assertEqual((tx["generation"], ty["generation"]), ("stream", "chunked"))
                        stats["stream"] += 1
                        stats[tx["ended"]] += 1
        for what in ("loop", "closed", "budget"):
            self.assertGreater(stats[what], 20, (what, stats))

    async def test_bursts_move_the_check_by_less_than_a_burst(self):
        """With several tokens per event (speculative decoding) the loop detector runs at the first event at or past
        1,024 / 2,048 / 3,072 tokens, at most burst - 1 tokens later than in chunks. Both cut by the same rule
        (loop_cut, a bisection to within 8 characters), so the cut can move by a few characters."""
        r = random.Random(7)
        compared, moved = 0, []
        folder = folder_for(random_settings(random.Random(0)))
        auto = replace(folder.auto_think, think_mode="always", budget=4096, chunk_tokens=1024, prefix="P\n",
                       tail_chars=0)
        folder = replace(folder, auto_think=auto)
        for case in range(120):
            question = {"q": {"type": "noul", "instructions": f"case {case} {r.random()}"}}
            out = {}
            for version, burst in (("0.31.0", 3), ("0.19.1", 1)):
                fake = FakeVLLM(version=version, burst=burst, kinds=("loop_short", "loop_lines", "loop_long", "digits"))
                b = VLLMBackend(["8001"], FakeTokenizer(), waves=False, transport=fake)
                await b.start()
                ctl = Controls(think_mode="always", budget=4096, details=True, thought_text=True)
                out[version] = (await Decider(folder, b).decide("x", question, ctl)).answers["q"]["details"]["thought"]
                await b.close()
            s, c = out["0.31.0"], out["0.19.1"]
            if c["ended"] == "loop" and c["tokens"] in (1024, 2048, 3072):
                compared += 1
                self.assertEqual(s["ended"], "loop")
                self.assertTrue(c["tokens"] <= s["tokens"] <= c["tokens"] + 2, (s["tokens"], c["tokens"]))
                short, long_ = sorted((s["text"], c["text"]), key=len)
                self.assertTrue(long_.startswith(short))
                moved.append(len(long_) - len(short))
        self.assertGreater(compared, 30)
        self.assertLessEqual(sorted(moved)[len(moved) // 2], 8, moved)     # usually within the bisection step
        print(f"\nbursts of 3 tokens: {compared} loops cut at the same check, the same cut in {moved.count(0)}, "
              f"median shift {sorted(moved)[len(moved) // 2]} characters, largest {max(moved)}")


if __name__ == "__main__":
    unittest.main()
