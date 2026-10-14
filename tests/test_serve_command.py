"""`startlux-decide serve` end to end: the command runs as its own process against a fake vLLM server, with the real
tokenizer of a model folder (only its tokenizer and config files are copied; never weights), once plainly and once
the way it is served publicly (API keys, HTTPS, in-flight limit, OpenRouter document, access log). Needs aiohttp,
transformers and STARTLUX_TEST_MODEL_DIR."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import unittest

from startlux_decide.quantize import FOLDER_FILES

from .certs import self_signed
from .fake_vllm import FakeVLLM
from .test_serving_files import KEY_A

MODEL_DIR = os.environ.get("STARTLUX_TEST_MODEL_DIR")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import aiohttp
    import transformers  # noqa: F401  (the server loads the folder's tokenizer)

    from .test_http_engines import FakeServer, vllm_handler
except ImportError:
    aiohttp = None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(aiohttp is not None and MODEL_DIR, "needs aiohttp, transformers and STARTLUX_TEST_MODEL_DIR")
class ServeCommandTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="startlux_serve_")
        self.folder = os.path.join(self.dir, "model")
        os.mkdir(self.folder)
        for name in FOLDER_FILES:
            if os.path.isfile(os.path.join(MODEL_DIR, name)):
                shutil.copy(os.path.join(MODEL_DIR, name), self.folder)
        self.port = free_port()

    async def asyncTearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    async def serve(self, vllm_url: str, *extra: str):
        """Start `serve`; return the process once /health answers."""
        proc = subprocess.Popen([sys.executable, "-m", "startlux_decide.cli", "serve", "--model", self.folder,
                                 "--vllm-url", vllm_url, "--port", str(self.port), "--name", "test-27b",
                                 "--wave-gather", "0.01", *extra],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc

    @staticmethod
    async def stop(proc) -> str:
        """Stop `serve` (SIGTERM: a graceful shutdown) and return what it printed."""
        proc.terminate()
        return (await asyncio.get_running_loop().run_in_executor(None, lambda: proc.communicate(timeout=30)))[0]

    async def wait_health(self, http, proc, base: str, **kw) -> dict:
        for _ in range(600):
            try:
                async with http.get(base + "/health", **kw) as r:
                    return await r.json()
            except aiohttp.ClientError:
                if proc.poll() is not None:
                    self.fail("serve exited:\n" + proc.stdout.read())
                await asyncio.sleep(0.1)
        self.fail("serve did not come up")

    async def test_serve(self):
        for version, generation in (("0.19.1", "chunked"), ("0.31.0", "stream")):
            with self.subTest(vllm=version):
                await self.serve_once(version, generation)

    async def serve_once(self, version: str, generation: str):
        async with FakeServer(vllm_handler(FakeVLLM(version=version))) as vllm, aiohttp.ClientSession() as http:
            proc = await self.serve(vllm.url, "--max-think-budget", "8192")
            try:
                base = f"http://127.0.0.1:{self.port}"
                health = await self.wait_health(http, proc, base)
                self.assertEqual((health["model"], health["engine"]["served_model"], health["context_length"]),
                                 ("test-27b", "m", 65536))
                if health["kind"] == "auto_think":
                    self.assertEqual(health["defaults"]["thought_generation"], generation)
                    self.assertEqual(health["engine"]["vllm_versions"], [version])
                body = {"state": {"ticket": "charged twice"}, "think_mode": "always", "think_budget": 6000,
                        "return_thought": True, "questions": {
                            "team": {"type": "choice", "instructions": "Which team?",
                                     "criteria": {"billing": "Payments", "shipping": "Delivery"}},
                            "urgent": {"type": "noul", "instructions": "Answer today?"}}}
                if health["kind"] == "instant":
                    body["think_mode"] = "never"
                async with http.post(base + "/v1/systemone", json=body) as r:
                    self.assertEqual(r.status, 200, await r.text())
                    out = await r.json()
                self.assertEqual(out["model"], "test-27b")
                self.assertEqual(set(out["answers"]), {"team", "urgent"})
                self.assertIn("timing_ms", out)
                thought = out["answers"]["team"]["details"]["thought"]
                self.assertEqual(thought["used"], health["kind"] == "auto_think")
                if thought["used"]:
                    self.assertTrue(thought["text"].startswith("Let me work through this carefully"))
                    self.assertLessEqual(thought["tokens"], 6000)
                    self.assertEqual(thought["generation"], generation)
            finally:
                await self.stop(proc)

    async def test_serve_publicly(self):
        certs = self_signed(self.dir)
        if certs is None:
            self.skipTest("openssl is not available")
        keys, log = os.path.join(self.dir, "keys"), os.path.join(self.dir, "access.jsonl")
        with open(keys, "w") as f:
            f.write(f"# OpenRouter\n{KEY_A}\n")
        models = os.path.join(HERE, "docs", "openrouter_models.example.json")
        tls = ssl.create_default_context(cafile=certs[0])
        tls.check_hostname = False
        async with FakeServer(vllm_handler(FakeVLLM())) as vllm, \
                aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as http:
            proc = await self.serve(vllm.url, "--api-keys-file", keys, "--ssl-certfile", certs[0], "--ssl-keyfile",
                                    certs[1], "--max-inflight", "4", "--openrouter-models-file", models,
                                    "--access-log", log)
            try:
                base = f"https://127.0.0.1:{self.port}"
                await self.wait_health(http, proc, base, ssl=tls)
                body = {"model": "startlux/startlux-decision-27b-autothink", "state": "STATE-MARKER-9d1e",
                        "questions": {"q": {"type": "noul", "instructions": "QUESTION-MARKER-4c2a"}}}
                async with http.post(base + "/v1/decisions", json=body, ssl=tls) as r:
                    self.assertEqual(r.status, 401)
                auth = {"Authorization": f"Bearer {KEY_A}"}
                async with http.post(base + "/v1/decisions", json=body, ssl=tls, headers=auth) as r:
                    self.assertEqual(r.status, 200, await r.text())
                    self.assertEqual((await r.json())["model"], "test-27b")
                async with http.get(base + "/openrouter/models", ssl=tls, headers=auth) as r:
                    with open(models, "rb") as f:
                        self.assertEqual(await r.read(), f.read())
            finally:
                output = await self.stop(proc)
        with open(log) as f:
            text = f.read()
        self.assertEqual([json.loads(x)["status"] for x in text.splitlines()], [200, 401, 200, 200])
        for secret in ("STATE-MARKER-9d1e", "QUESTION-MARKER-4c2a", KEY_A):
            self.assertNotIn(secret, text + output)
        self.assertIn("https://", output)


if __name__ == "__main__":
    unittest.main()
