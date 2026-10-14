"""The command line without engines: launch-vllm --dry-run and the quantize steps."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from startlux_decide import cli, quantize

from .folders import make_folder

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.auto, self.inst, self.tmp = make_folder(), make_folder("instant"), tempfile.mkdtemp()

    def tearDown(self):
        for p in (self.auto, self.inst, self.tmp):
            shutil.rmtree(p, ignore_errors=True)

    def test_launch_dry_run(self):
        code, out, _ = run_cli("launch-vllm", "--model", self.inst, "--gpus", "2,3", "--python", "py", "--dry-run")
        lines = out.strip().splitlines()
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("CUDA_VISIBLE_DEVICES=2 STARTLUX_PREFIX_BIDIR=1 VLLM_NO_USAGE_STATS=1 "
                                            "py -m vllm.entrypoints.openai.api_server"))
        self.assertIn("--port 8002", lines[1])
        self.assertIn("--limit-mm-per-prompt '{\"image\":0,\"video\":0}'", lines[0])

    def test_launch_dry_run_on_vllm_031(self):
        with open(os.path.join(self.auto, "config.json"), "w") as f:
            json.dump({"text_config": {"mtp_num_hidden_layers": 1}}, f)
        with open(os.path.join(self.auto, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"mtp.fc.weight": "mtp.safetensors"}}, f)
        with mock.patch("startlux_decide.launch.vllm_version", return_value="0.31.0"):
            code, out, err = run_cli("launch-vllm", "--model", self.auto, "--python", "py", "--dry-run")
            self.assertEqual(code, 0)
            self.assertIn("--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192", out)
            self.assertIn("--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'", out)
            self.assertIn("MTP speculative decoding with 3 draft tokens", err)
            code, out, err = run_cli("launch-vllm", "--model", self.auto, "--python", "py", "--dry-run", "--no-mtp")
            self.assertNotIn("--speculative-config", out)
            self.assertIn("--enable-prefix-caching", out)
            code, out, _ = run_cli("launch-vllm", "--model", self.inst, "--python", "py", "--dry-run")
            self.assertIn("--no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 65536", out)

    def test_quantize(self):
        code, out, _ = run_cli("quantize", "fp8", "--model", self.auto)
        self.assertIn("--quantization fp8", out)
        open(os.path.join(self.tmp, "convert_hf_to_gguf.py"), "w").close()        # a stock converter: Instant refused
        code, out, err = run_cli("quantize", "gguf", "--model", self.inst, "--out", self.tmp, "--llama-cpp", self.tmp)
        self.assertIn("0002-convert-qwen35-prompt-bidirectional", err)
        os.makedirs(os.path.join(self.tmp, "conversion"), exist_ok=True)       # the patched converter: accepted
        with open(os.path.join(self.tmp, "conversion", "qwen.py"), "w") as f:
            f.write('if self.hparams.get("jev_prompt_attention") == "bidirectional": pass\n')
        code, out, err = run_cli("quantize", "gguf", "--model", self.inst, "--out", self.tmp, "--llama-cpp", self.tmp)
        self.assertNotIn("0002-convert", err)
        os.remove(os.path.join(self.tmp, "convert_hf_to_gguf.py"))
        code, out, _ = run_cli("quantize", "mlx", "--model", self.inst, "--out", self.tmp, "--bits", "8")
        self.assertIn("mlx_vlm convert --hf-path " + self.inst, out)
        self.assertIn("-q --q-bits 8 --q-group-size 64", out)
        code, out, err = run_cli("quantize", "gguf", "--model", self.auto, "--out", self.tmp, "--llama-cpp", self.tmp,
                                 "--type", "Q4_K_M")
        self.assertEqual(code, 0)
        self.assertIn("convert_hf_to_gguf.py " + self.auto + " --outtype bf16", out)
        self.assertIn("model-Q4_K_M.gguf Q4_K_M", out)
        self.assertIn("missing:", err)
        code, _, _ = run_cli("quantize", "gguf", "--model", self.auto, "--out", self.tmp, "--llama-cpp", self.tmp,
                             "--run")
        self.assertEqual(code, 1)                                   # tools missing: nothing is run

    def test_public_options(self):
        keys = os.path.join(self.tmp, "keys")
        with open(keys, "w") as f:
            f.write("# OpenRouter\nkey-0123456789abcdef-secret\n")
        args = cli.build_parser().parse_args(["serve", "--model", self.auto, "--api-keys-file", keys,
                                              "--max-inflight", "16", "--access-log",
                                              os.path.join(self.tmp, "access.jsonl"), "--openrouter-models-file",
                                              os.path.join(HERE, "docs", "openrouter_models.example.json")])
        options, tls = cli._public_options(args)
        self.assertIsNone(tls)
        self.assertEqual((options.max_inflight, options.models_path), (16, "/openrouter/models"))
        self.assertTrue(options.api_keys.check("Bearer key-0123456789abcdef-secret"))
        options.access_log.close()
        for argv, fragment in ((["--ssl-certfile", "c.pem"], "go together"),
                               (["--openrouter-models-path", "/v1/systemone"], "must start with /"),
                               (["--api-keys-file", os.path.join(self.tmp, "missing")], "cannot serve"),
                               (["--openrouter-models-file", keys], "not valid JSON")):
            with self.assertRaises(SystemExit) as cm:
                cli._public_options(cli.build_parser().parse_args(["serve", "--model", self.auto] + argv))
            self.assertIn(fragment, str(cm.exception))
        with open(keys, "w") as f:
            f.write("key-0123456789abcdef-secret extra\n")
        with self.assertRaises(SystemExit) as cm:
            cli._public_options(cli.build_parser().parse_args(["serve", "--model", self.auto, "--api-keys-file", keys]))
        self.assertIn("line 1", str(cm.exception))
        self.assertNotIn("secret", str(cm.exception))
        with self.assertRaises(SystemExit):
            run_cli("serve", "--model", self.auto, "--max-inflight", "0")

    def test_copy_folder_files(self):
        out = os.path.join(self.tmp, "build")
        self.assertEqual(quantize.copy_folder_files(self.auto, out), ["config.json", "decision_config.json"])
        self.assertEqual(quantize.copy_folder_files(self.auto, out), [])


if __name__ == "__main__":
    unittest.main()
