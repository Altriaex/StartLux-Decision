"""API keys, files re-read when they change, the OpenRouter models document, the access log and the TLS context."""
from __future__ import annotations

import json
import logging
import os
import shutil
import ssl
import tempfile
import unittest

from startlux_decide.serving import (AccessLog, ApiKeys, WatchedFile, parse_keys, parse_models_document,
                                     tls_context)

from .certs import self_signed

KEY_A = "key-A-7f3c9e1d5b2a4c6e8f0a"
KEY_B = "key-B-0d2f4b6a8c1e3a5c7e9b"


def replace_file(path: str, text: str) -> None:
    """Write a new version of a file the way operators should: a new file renamed over the old one."""
    tmp = path + ".new"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class KeysTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "keys")
        replace_file(self.path, f"# clients\n\n{KEY_A}\n   # indented comment\n  {KEY_B}  \n")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_parse(self):
        with open(self.path, "rb") as f:
            self.assertEqual(len(parse_keys(f.read())), 2)
        for bad, fragment in ((b"short-key\n", "line 1: a key must have at least 16"),
                              (b"# c\nkey with spaces in it\n", "line 2: a key must be one word"),
                              ("kéy-with-accent-0123".encode(), "line 1"), (b"\xff\xfe", "not UTF-8")):
            with self.assertRaises(ValueError) as cm:
                parse_keys(bad)
            self.assertIn(fragment, str(cm.exception))
            self.assertNotIn("spaces in it", str(cm.exception))
        self.assertEqual(parse_keys(b"# nothing but a comment\n"), frozenset())

    def test_check(self):
        keys = ApiKeys(self.path)
        for header, ok in ((f"Bearer {KEY_A}", True), (f"bearer {KEY_B}", True), (f"Bearer  {KEY_A} ", True),
                           (None, False), ("", False), (KEY_A, False), (f"Basic {KEY_A}", False), ("Bearer ", False),
                           (f"Bearer {KEY_A}x", False), (f"Bearer {KEY_A[:-1]}", False), ("Bearer \udcff", False)):
            self.assertEqual(keys.check(header), ok, header)

    def test_reload(self):
        keys = ApiKeys(self.path)
        replace_file(self.path, KEY_B + "\n")
        self.assertFalse(keys.check(f"Bearer {KEY_A}"))
        self.assertTrue(keys.check(f"Bearer {KEY_B}"))
        with self.assertLogs("startlux_decide.serving", "ERROR") as logs:       # a bad version is not applied
            replace_file(self.path, f"{KEY_A}\nnot a key\n")
            self.assertTrue(keys.check(f"Bearer {KEY_B}"))
            self.assertFalse(keys.check(f"Bearer {KEY_A}"))
        self.assertIn("line 2", logs.output[0])
        self.assertNotIn(KEY_A, "".join(logs.output))
        replace_file(self.path, "# all keys revoked\n")                         # an empty file locks everyone out
        self.assertFalse(keys.check(f"Bearer {KEY_B}"))

    def test_start_needs_keys(self):
        replace_file(self.path, "# none yet\n")
        with self.assertRaisesRegex(ValueError, "no keys"):
            ApiKeys(self.path)
        with self.assertRaises(OSError):
            ApiKeys(os.path.join(self.dir, "missing"))

    def test_no_key_in_any_log(self):
        root = logging.getLogger()
        records = []
        handler = logging.Handler(logging.DEBUG)
        handler.emit = lambda r: records.append(r.getMessage())
        root.addHandler(handler)
        try:
            keys = ApiKeys(self.path)
            replace_file(self.path, f"{KEY_A} {KEY_B}\n")
            keys.check(f"Bearer {KEY_A}")
            os.remove(self.path)
            keys.check(f"Bearer {KEY_B}")
        finally:
            root.removeHandler(handler)
        self.assertTrue(records)
        self.assertFalse([r for r in records if KEY_A in r or KEY_B in r])


class ModelsDocumentTest(unittest.TestCase):
    def test_example_document_is_valid(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "docs", "openrouter_models.example.json"), "rb") as f:
            raw = f.read()
        self.assertIs(parse_models_document(raw), raw)
        model = json.loads(raw)["data"][0]
        for field in ("id", "name", "created", "description", "schema_version", "input_modalities",
                      "output_modalities", "quantization", "datacenters", "is_free", "capacity"):
            self.assertIn(field, model)

    def test_checks(self):
        for bad in (b"{", b"[]", b'{"data": []}', b'{"data": [{"name": "x"}]}', b'{"data": [{"id": ""}]}'):
            with self.assertRaises(ValueError):
                parse_models_document(bad)

    def test_watched_reload(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "models.json")
            replace_file(path, '{"data": [{"id": "a/b"}]}')
            doc = WatchedFile(path, parse_models_document, "models")
            self.assertEqual(doc.get(), b'{"data": [{"id": "a/b"}]}')
            replace_file(path, '{"data": [{"id": "a/c"}]}\n')
            self.assertEqual(doc.get(), b'{"data": [{"id": "a/c"}]}\n')
            with self.assertLogs("startlux_decide.serving", "ERROR"):
                replace_file(path, "{broken")
                self.assertEqual(doc.get(), b'{"data": [{"id": "a/c"}]}\n')
        finally:
            shutil.rmtree(d)


class AccessLogAndTlsTest(unittest.TestCase):
    def test_access_log_lines(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "access.jsonl")
            log = AccessLog(path)
            log.write({"request_id": "r1", "status": 200})
            log.write({"request_id": "r2", "status": 429})
            log.close()
            with open(path) as f:
                self.assertEqual([json.loads(x)["status"] for x in f], [200, 429])
        finally:
            shutil.rmtree(d)

    def test_tls_context(self):
        d = tempfile.mkdtemp()
        try:
            certs = self_signed(d)
            if certs is None:
                self.skipTest("openssl is not available")
            ctx = tls_context(*certs)
            self.assertEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
            with self.assertRaises(OSError):
                tls_context(os.path.join(d, "missing.pem"), certs[1])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
