"""Long states: the questions of a request that share a long state read it once on an engine with a prefix cache
(one prefill of the shared beginning, then every question on that server), and not on engines or folders without
one; a prompt longer than the engine's context is refused with the engine's reason."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.backends.vllm import VLLMBackend
from startlux_decide.controls import Controls, RequestError
from startlux_decide.pipeline import SHARE_MIN, Decider
from startlux_decide.prompt import count_tokens

from .fake_vllm import FakeVLLM
from .test_pipeline import AUTO, QUESTIONS

INSTANT = replace(AUTO, prompt_attention="bidirectional", auto_think=None)
LONG = "Ticket history. " * 600                     # about 9,600 tokens for the fake tokenizer


class SharedStateTest(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, version: str, folder=AUTO, state: str = LONG, questions=None, strict=False):
        fakes = [FakeVLLM(version=version, kinds=("close",), strict=strict, max_model_len=20000) for _ in range(2)]
        servers = {f"http://127.0.0.1:{8001 + i}": f for i, f in enumerate(fakes)}

        async def transport(url, body):
            base = url[:url.index("/", len("http://"))]
            return await servers[base](url, body)
        backend = VLLMBackend(list(servers), FakeTokenizer(), transport=transport)
        await backend.start()
        logs = [len(f.log) for f in fakes]
        out = await Decider(folder, backend).decide(state, questions or QUESTIONS, Controls(think_mode="never"))
        return out, fakes, [f.log[n:] for f, n in zip(fakes, logs)]

    async def test_long_state_is_read_once(self):
        out, fakes, logs = await self.run_request("0.31.0")
        used = [i for i, log in enumerate(logs) if log]
        self.assertEqual(len(used), 1)                                  # every call on one server
        fake = fakes[used[0]]
        self.assertEqual(len(fake.prefills), 1)                         # the shared beginning, once, first
        prefix = fake.prefills[0][0]
        self.assertGreaterEqual(count_tokens(FakeTokenizer(), prefix), SHARE_MIN)
        bodies = [json.loads(x) for x in logs[used[0]]]
        self.assertEqual((bodies[0]["prompt"], bodies[0]["max_tokens"]), (prefix, 1))
        self.assertEqual(len(bodies), 4)                                # then the three read-outs
        self.assertTrue(all(b["prompt"].startswith(prefix) and "logprobs" in b for b in bodies[1:]))
        self.assertIn(LONG.strip(), prefix)

    async def test_no_sharing_without_a_cache_or_for_instant(self):
        for version, folder, state in (("0.19.1", AUTO, LONG), ("0.31.0", INSTANT, LONG), ("0.31.0", AUTO, "short")):
            with self.subTest(version=version, folder=folder.kind, state=len(state)):
                _, fakes, logs = await self.run_request(version, folder, state)
                self.assertEqual(sum(len(f.prefills) for f in fakes), 0)
                if version == "0.19.1":                             # questions spread round robin
                    self.assertEqual(sorted(len(log) for log in logs), [1, 2])
        _, fakes, _ = await self.run_request("0.31.0", questions={"q": QUESTIONS["urgent"]})
        self.assertEqual(sum(len(f.prefills) for f in fakes), 0)        # one question: nothing to share

    async def test_prompt_over_the_context(self):
        with self.assertRaises(RequestError) as cm:
            await self.run_request("0.31.0", state="x" * 30000, strict=True, questions={"q": QUESTIONS["urgent"]})
        self.assertIn("maximum context length is 20000 tokens", str(cm.exception))
        self.assertEqual(cm.exception.loc, ["body", "questions", "q"])


if __name__ == "__main__":
    unittest.main()
