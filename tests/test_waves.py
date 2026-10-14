"""Wave batching of thought requests: the package's WaveScheduler groups requests exactly as the release proxy's
WaveGen did, and a caller that gives up does not stop the worker."""
from __future__ import annotations

import asyncio
import unittest

from startlux_decide.backends.vllm import WaveScheduler

from . import reference_impl as ref


class WaveTest(unittest.TestCase):
    def test_same_waves_as_the_release_proxy(self):
        def ours(send):
            return WaveScheduler(send, max_wave=128, gather_s=0.05)

        def release(send):
            proxy = type("P", (), {"complete": staticmethod(lambda url, body: send(body))})()
            return ref.WaveGen(proxy, "url", 128, 0.05)

        got, want = asyncio.run(self._grouped(ours)), asyncio.run(self._grouped(release))
        self.assertEqual(got, want)
        self.assertEqual(got, [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10]])

    @staticmethod
    async def _grouped(make):
        started = []

        async def send(body):
            started.append((asyncio.get_running_loop().time(), body["n"]))
            await asyncio.sleep(0.2)
            return {"n": body["n"]}

        sched = make(send)

        async def submit(n, delay):
            await asyncio.sleep(delay)
            return (await sched.submit({"n": n}))["n"]

        jobs = [submit(n, 0.0) for n in range(6)] + [submit(n, 0.1) for n in range(6, 9)] + \
               [submit(n, 0.6) for n in (9, 10)]
        results = await asyncio.gather(*jobs)
        assert results == list(range(11))
        waves, last = [], None
        for t, n in sorted(started):
            if last is None or t - last > 0.1:
                waves.append([])
            waves[-1].append(n)
            last = t
        return [sorted(w) for w in waves]

    def test_max_wave(self):
        async def go():
            sizes = []
            in_flight = [0]

            async def send(body):
                in_flight[0] += 1
                sizes.append(in_flight[0])
                await asyncio.sleep(0.05)
                in_flight[0] -= 1
                return body

            sched = WaveScheduler(send, max_wave=4, gather_s=0.01)
            await asyncio.gather(*(sched.submit({"n": n}) for n in range(10)))
            await sched.close()
            return max(sizes)
        self.assertEqual(asyncio.run(go()), 4)

    def test_cancelled_caller(self):
        async def go():
            async def send(body):
                await asyncio.sleep(0.05)
                return body

            sched = WaveScheduler(send, gather_s=0.02)
            first = asyncio.ensure_future(sched.submit({"n": 1}))
            await asyncio.sleep(0.03)
            first.cancel()                                  # the client went away during the wave
            await asyncio.sleep(0.1)
            out = await sched.submit({"n": 2})              # the worker still serves later waves
            await sched.close()
            return out
        self.assertEqual(asyncio.run(go()), {"n": 2})


if __name__ == "__main__":
    unittest.main()
