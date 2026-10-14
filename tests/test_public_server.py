"""The server's options for serving publicly: API keys (401 / 200, reload), early 429s, /v1/decisions, the OpenRouter
models document, HTTPS, and an access log and server log without content. Needs aiohttp."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import socket
import ssl
import stat
import tempfile
import time
import unittest
import warnings

from startlux_decide.backends.fake import FakeBackend
from startlux_decide.modelcfg import load_model_folder
from startlux_decide.serving import (AccessLog, ApiKeys, RequestLog, ServeOptions, WatchedFile, parse_models_document,
                                     tls_context)

from .certs import self_signed
from .folders import make_folder
from .test_images import Image, png
from .test_pipeline import closing
from .test_serving_files import KEY_A, KEY_B, replace_file

try:
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from startlux_decide.server import create_app
except ImportError:                       # the pure-logic suite runs without aiohttp
    aiohttp = None

MARKERS = ("STATE-MARKER-1c9e", "QUESTION-MARKER-77ab", "OPTION-MARKER-3f02")
BODY = {"state": {"note": MARKERS[0]}, "questions": {
    "pick": {"type": "choice", "instructions": f"Which one? {MARKERS[1]}",
             "criteria": {MARKERS[2]: "first", "b": None}},
    "late": {"type": "noul", "instructions": "Late?"}}}


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


class Recorder(logging.Handler):
    """Every log message of the process while a test runs."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@unittest.skipIf(aiohttp is None, "needs aiohttp")
class PublicServerTest(unittest.IsolatedAsyncioTestCase):
    delay = 0.0

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp()
        self.folder = make_folder()
        self.keys_path = os.path.join(self.dir, "keys")
        replace_file(self.keys_path, f"# clients\n{KEY_A}\n")
        self.models_path = os.path.join(self.dir, "models.json")
        replace_file(self.models_path, '{"data": [{"id": "startlux/test", "is_free": true}]}\n')
        self.log_path = os.path.join(self.dir, "access.jsonl")
        self.recorder = Recorder()
        logging.getLogger().addHandler(self.recorder)
        logging.getLogger("startlux_decide").setLevel(logging.DEBUG)
        self.options = ServeOptions(
            api_keys=ApiKeys(self.keys_path), max_inflight=self.max_inflight(),
            models_document=WatchedFile(self.models_path, parse_models_document, "OpenRouter models document"),
            access_log=AccessLog(self.log_path), request_log=self.make_request_log())
        app = create_app(load_model_folder(self.folder), FakeBackend(script=closing, delay=self.delay), "test-model",
                         options=self.options)
        server = TestServer(app)
        await server.start_server(access_log=None)          # as `serve` runs it: no aiohttp access log
        self.client = TestClient(server)

    def max_inflight(self):
        return None

    def make_request_log(self):
        return None

    async def asyncTearDown(self):
        await self.client.close()
        logging.getLogger().removeHandler(self.recorder)
        logging.getLogger("startlux_decide").setLevel(logging.NOTSET)
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.folder, ignore_errors=True)

    async def post(self, path="/v1/systemone", body=None, headers=None):
        r = await self.client.post(path, json=BODY if body is None else body, headers=headers)
        return r, await r.json()

    def no_secrets_anywhere(self):
        with open(self.log_path) as f:
            access = f.read()
        everything = access + "\n".join(self.recorder.messages)
        for secret in MARKERS + (KEY_A, KEY_B):
            self.assertNotIn(secret, everything)
        return [json.loads(line) for line in access.splitlines()]


class KeysTest(PublicServerTest):
    async def test_401_and_200(self):
        for headers in (None, bearer(KEY_B), {"Authorization": KEY_A}, {"Authorization": f"Basic {KEY_A}"}):
            r, out = await self.post(headers=headers)
            self.assertEqual(r.status, 401)
            self.assertEqual(out["detail"][0]["type"], "unauthorized")
            self.assertEqual(r.headers["WWW-Authenticate"], "Bearer")
            self.assertNotIn(KEY_A, json.dumps(out))
        for path in ("/v1/systemone", "/v1/decisions"):
            r, out = await self.post(path, headers=bearer(KEY_A))
            self.assertEqual(r.status, 200, out)
        for path in ("/v1/models", "/openrouter/models", "/nope"):
            self.assertEqual((await self.client.get(path)).status, 401)
        self.assertEqual((await self.client.get("/nope", headers=bearer(KEY_A))).status, 404)
        for path in ("/health", "/v1/health"):
            r = await self.client.get(path)
            self.assertEqual(r.status, 200)
            self.assertEqual(set(await r.json()), {"status", "model", "kind"})    # no engine addresses without a key
            r = await self.client.get(path, headers=bearer(KEY_A))
            self.assertIn("engine", await r.json())
        self.no_secrets_anywhere()

    async def test_key_file_is_read_again_when_it_changes(self):
        self.assertEqual((await self.post(headers=bearer(KEY_A)))[0].status, 200)
        replace_file(self.keys_path, f"{KEY_B}\n")
        self.assertEqual((await self.post(headers=bearer(KEY_A)))[0].status, 401)
        self.assertEqual((await self.post(headers=bearer(KEY_B)))[0].status, 200)
        replace_file(self.keys_path, f"{KEY_A}\nbad key line\n")          # not applied: B stays valid
        self.assertEqual((await self.post(headers=bearer(KEY_B)))[0].status, 200)
        self.assertEqual((await self.post(headers=bearer(KEY_A)))[0].status, 401)
        replace_file(self.keys_path, "# revoked\n")
        self.assertEqual((await self.post(headers=bearer(KEY_B)))[0].status, 401)
        self.assertTrue(any("line 2" in m for m in self.recorder.messages))
        self.no_secrets_anywhere()


class DecisionsAndModelsTest(PublicServerTest):
    async def test_decisions_is_systemone(self):
        body = dict(BODY, model="openrouter/some-other-name", return_details=True, think_mode="always")
        r1, a = await self.post("/v1/systemone", body, bearer(KEY_A))
        r2, b = await self.post("/v1/decisions", body, bearer(KEY_A))
        self.assertEqual((r1.status, r2.status), (200, 200))
        self.assertEqual(a["model"], "test-model")
        strip = lambda out: {k: {f: v for f, v in ans.items() if f != "details"}      # noqa: E731
                             for k, ans in out["answers"].items()}
        self.assertEqual(strip(a), strip(b))
        self.assertEqual(a["usage"], b["usage"])

    async def test_models_document(self):
        r = await self.client.get("/openrouter/models", headers=bearer(KEY_A))
        self.assertEqual((r.status, r.content_type), (200, "application/json"))
        self.assertEqual(await r.read(), b'{"data": [{"id": "startlux/test", "is_free": true}]}\n')
        replace_file(self.models_path, '{"data": [{"id": "startlux/test-2"}]}')
        r = await self.client.get("/openrouter/models", headers=bearer(KEY_A))
        self.assertEqual(await r.read(), b'{"data": [{"id": "startlux/test-2"}]}')
        replace_file(self.models_path, '{"data": "broken"}')               # not applied
        r = await self.client.get("/openrouter/models", headers=bearer(KEY_A))
        self.assertEqual(await r.read(), b'{"data": [{"id": "startlux/test-2"}]}')


class EarlyLimitTest(PublicServerTest):
    delay = 0.5

    def max_inflight(self):
        return 2

    async def test_429_at_once(self):
        body = dict(BODY, think_mode="never")

        async def one():
            t0 = time.perf_counter()
            r = await self.client.post("/v1/decisions", json=body, headers=bearer(KEY_A))
            out = await r.json()
            return r.status, time.perf_counter() - t0, r.headers.get("Retry-After"), out

        results = await asyncio.gather(*(one() for _ in range(6)))
        statuses = sorted(s for s, _, _, _ in results)
        self.assertEqual(statuses, [200, 200, 429, 429, 429, 429])
        for status, seconds, retry, out in results:
            if status == 429:
                self.assertLess(seconds, 0.3)                       # answered at once, not queued
                self.assertEqual(retry, "1")
                self.assertEqual(out["detail"][0]["type"], "too_many_requests")
            else:
                self.assertGreaterEqual(seconds, 0.5)
        self.assertEqual((await self.client.get("/health")).status, 200)     # health is never limited
        r, _ = await self.post("/v1/decisions", body, bearer(KEY_A))
        self.assertEqual(r.status, 200)                             # capacity is back
        lines = self.no_secrets_anywhere()
        self.assertEqual(sorted(x["status"] for x in lines if x["route"] == "/v1/decisions"), [200] * 3 + [429] * 4)


class AccessLogTest(PublicServerTest):
    async def test_lines_without_content(self):
        await self.post("/v1/systemone", dict(BODY, return_details=True), bearer(KEY_A))
        await self.post("/v1/decisions", BODY, bearer(KEY_A))
        await self.post("/v1/systemone", BODY)                                      # 401
        bad = {"state": MARKERS[0], "questions": {"s": {"type": "score", "instructions": MARKERS[1],
                                                        "criteria": [MARKERS[2]] * 11}}}
        await self.post("/v1/systemone", bad, bearer(KEY_A))                        # 422
        await self.client.get("/health")
        await self.client.get("/nope?api_key=" + KEY_A, headers=bearer(KEY_A))       # 404, query not logged
        lines = self.no_secrets_anywhere()
        self.assertEqual([x["status"] for x in lines], [200, 200, 401, 422, 200, 404])
        fields = {"time", "request_id", "method", "route", "status", "total_ms", "client_ip", "key_id", "questions",
                  "input_tokens", "thought_tokens", "wall_ms", "first_pass_sum_ms", "thought_span_ms",
                  "after_thought_sum_ms"}
        self.assertTrue(all(set(x) == fields for x in lines))
        key_a = hashlib.sha256(KEY_A.encode()).hexdigest()[:16]
        self.assertEqual([x["key_id"] for x in lines], [key_a, key_a, None, key_a, None, key_a])
        self.assertEqual(lines[0]["client_ip"], "127.0.0.1")
        for x in lines[:2]:
            self.assertEqual(x["questions"], 2)
            self.assertGreater(x["input_tokens"], 0)
            self.assertIsInstance(x["thought_tokens"], int)
            self.assertIsNotNone(x["first_pass_sum_ms"])               # with or without return_details
        self.assertEqual([x["route"] for x in lines], ["/v1/systemone", "/v1/decisions", "/v1/systemone",
                                                       "/v1/systemone", "/health", None])
        self.assertIsNone(lines[2]["questions"])
        self.assertEqual(lines[3]["questions"], 1)
        self.assertTrue(lines[0]["time"].endswith("Z"))


class RequestLogTest(PublicServerTest):
    """--request-log: the content of every decision request, who sent it with which key, and the full reply."""

    def make_request_log(self):
        self.content_dir = os.path.join(self.dir, "requests")
        return RequestLog(self.content_dir)

    def lines(self):
        out = []
        for name in sorted(os.listdir(self.content_dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(self.content_dir, name), encoding="utf-8") as f:
                    out += [json.loads(x) for x in f]
        return out

    async def test_content_client_and_key(self):
        headers = dict(bearer(KEY_A), **{"X-Forwarded-For": "203.0.113.7, 198.51.100.2", "User-Agent": "probe/1"})
        r, plain = await self.post("/v1/decisions", dict(BODY, think_mode="always"), headers)
        self.assertEqual(r.status, 200, plain)
        self.assertNotIn("details", plain["answers"]["pick"])                # the client gets what it asked for
        self.assertNotIn("timing_ms", plain)
        await self.post("/v1/systemone", BODY)                                 # 401: refused before the body
        r, detailed = await self.post("/v1/systemone", dict(BODY, think_mode="always", return_details=True),
                                      bearer(KEY_A))
        self.assertNotIn("text", detailed["answers"]["pick"]["details"]["thought"])
        await self.client.post("/v1/systemone", data=f"not json {MARKERS[0]}", headers=bearer(KEY_A))   # 422
        lines = self.lines()
        self.assertEqual([x["status"] for x in lines], [200, 401, 200, 422])
        first = lines[0]
        self.assertEqual(first["request"]["state"], {"note": MARKERS[0]})
        self.assertEqual(first["client"]["ip"], "203.0.113.7")
        self.assertEqual(first["client"]["x_forwarded_for"], "203.0.113.7, 198.51.100.2")
        self.assertEqual((first["client"]["peer"], first["client"]["user_agent"]), ("127.0.0.1", "probe/1"))
        self.assertEqual(first["key_id"], hashlib.sha256(KEY_A.encode()).hexdigest()[:16])
        thought = first["response"]["answers"]["pick"]["details"]["thought"]
        self.assertTrue(thought["used"])
        self.assertTrue(thought["text"])                                      # the log keeps the thought
        self.assertIn("timing_ms", first["response"])
        self.assertEqual(plain["answers"]["pick"]["probabilities"],
                         first["response"]["answers"]["pick"]["probabilities"])
        self.assertIsNone(lines[1]["request"])
        self.assertIsNone(lines[1]["key_id"])
        self.assertEqual(lines[1]["response"]["detail"][0]["type"], "unauthorized")
        self.assertIn("text", lines[2]["response"]["answers"]["pick"]["details"]["thought"])
        self.assertEqual(lines[3]["request"], {"unparsed": f"not json {MARKERS[0]}"})
        self.assertEqual(stat.S_IMODE(os.stat(self.content_dir).st_mode), 0o700)
        for name in os.listdir(self.content_dir):
            if name.endswith(".jsonl"):
                path = os.path.join(self.content_dir, name)
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
                with open(path, encoding="utf-8") as f:
                    self.assertNotIn(KEY_A, f.read())                          # never the key itself
        with open(self.log_path) as f:
            self.assertNotIn(MARKERS[0], f.read())                             # the access log stays without content

    @unittest.skipIf(Image is None, "needs Pillow")
    async def test_images_are_stored_once(self):
        img = png(64, 48)
        await self.post("/v1/decisions", dict(BODY, images=[img, img]), bearer(KEY_A))
        refs = self.lines()[-1]["request"]["images"]
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0], refs[1])
        self.assertEqual((refs[0]["mime"], len(refs[0]["sha256"])), ("image/png", 64))
        path = os.path.join(self.content_dir, refs[0]["file"])
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        with open(path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), refs[0]["sha256"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipIf(aiohttp is None, "needs aiohttp")
class TlsTest(unittest.IsolatedAsyncioTestCase):
    async def test_https(self):
        d = tempfile.mkdtemp()
        folder = make_folder()
        try:
            certs = self_signed(d)
            if certs is None:
                self.skipTest("openssl is not available")
            runner = web.AppRunner(create_app(load_model_folder(folder), FakeBackend(), "test-model"))
            await runner.setup()
            port = free_port()
            await web.TCPSite(runner, "127.0.0.1", port, ssl_context=tls_context(*certs)).start()
            client_tls = ssl.create_default_context(cafile=certs[0])
            client_tls.check_hostname = False                       # the throwaway certificate has no names
            failed_handshakes = logging.getLogger("asyncio")       # the refused connections below are expected
            level = failed_handshakes.level
            failed_handshakes.setLevel(logging.CRITICAL)
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.post(f"https://127.0.0.1:{port}/v1/systemone", json=BODY, ssl=client_tls) as r:
                        self.assertEqual(r.status, 200)
                        self.assertIn("answers", await r.json())
                    _, writer = await asyncio.open_connection("127.0.0.1", port, ssl=client_tls)
                    self.assertIn(writer.get_extra_info("ssl_object").version(), ("TLSv1.2", "TLSv1.3"))
                    writer.close()
                    with self.assertRaises(aiohttp.ClientError):    # plain HTTP to the TLS port
                        async with http.get(f"http://127.0.0.1:{port}/health") as r:
                            await r.read()
                    old = ssl.create_default_context(cafile=certs[0])
                    old.check_hostname = False
                    try:
                        with warnings.catch_warnings():             # TLS 1.1 is deprecated: that is the point
                            warnings.simplefilter("ignore", DeprecationWarning)
                            old.maximum_version = ssl.TLSVersion.TLSv1_1
                    except (ValueError, ssl.SSLError):
                        return
                    with self.assertRaises((aiohttp.ClientError, ssl.SSLError)):   # no TLS 1.1 connection
                        async with http.get(f"https://127.0.0.1:{port}/health", ssl=old) as r:
                            await r.read()
            finally:
                await runner.cleanup()
                failed_handshakes.setLevel(level)
        finally:
            shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
