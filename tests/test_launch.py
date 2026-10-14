"""vLLM server commands, environments, the plugin check and the shutdown of the launcher (vLLM itself is not run);
the flags that depend on the vLLM version and the folder (prefix caching, chunked prefill, MTP)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from unittest import mock

from startlux_decide.launch import (VLLMServers, _signal_group, has_mtp_weights, plugin_problem, server_command,
                                    server_env)
from startlux_decide.vllm_plugin import ACTIVE_LINE

from .folders import make_folder

RELEASE_VALUES = [("--max-model-len", "65536"), ("--gpu-memory-utilization", "0.88"),
                  ("--logprobs-mode", "processed_logprobs"), ("--max-logprobs", "26"), ("--generation-config", "vllm"),
                  ("--max-num-batched-tokens", "65536"), ("--limit-mm-per-prompt", '{"image":0,"video":0}'),
                  ("--served-model-name", "m")]
RELEASE_SWITCHES = ["--no-enable-prefix-caching", "--no-enable-chunked-prefill"]


class CommandTest(unittest.TestCase):
    def test_release_flags(self):
        cmd = server_command("python3", "/models/m4", 8001)
        self.assertEqual(cmd[:5], ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/m4"])
        for flag, value in RELEASE_VALUES + [("--port", "8001"), ("--host", "127.0.0.1")]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)
        for switch in RELEASE_SWITCHES:
            self.assertIn(switch, cmd)
        self.assertNotIn("--quantization", cmd)
        self.assertEqual(server_command("p", "m", 1, quantization="fp8")[-2:], ["--quantization", "fp8"])

    def test_environment(self):
        env = server_env("3", True, {"PATH": "/bin"})
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["VLLM_NO_USAGE_STATS"], env["STARTLUX_PREFIX_BIDIR"]),
                         ("3", "1", "1"))
        self.assertNotIn("STARTLUX_PREFIX_BIDIR", server_env("0", False, {}))
        with self.assertRaisesRegex(RuntimeError, "VLLM_PLUGINS"):
            server_env("0", True, {"VLLM_PLUGINS": "other_plugin"})
        self.assertEqual(server_env("0", True, {"VLLM_PLUGINS": "other,startlux_prefix_bidir"})["VLLM_PLUGINS"],
                         "other,startlux_prefix_bidir")

    def test_plugin_log_check(self):
        self.assertIsNone(plugin_problem(f"... [startlux prefix-bidir pid=7] {ACTIVE_LINE} for /m ...\n"))
        self.assertIn("NOT installed", plugin_problem("[startlux prefix-bidir pid=7] NOT installed: written for vLLM "
                                                      "0.19.1, found 0.20.0\n"))
        self.assertIn("did not load the plugin", plugin_problem("INFO vLLM API server started\n"))

    def test_servers_plan(self):
        inst, auto = make_folder("instant"), make_folder()
        try:
            s = VLLMServers(inst, ["0", "1"], 9001, python="py")
            self.assertEqual(s.urls, ["http://127.0.0.1:9001", "http://127.0.0.1:9002"])
            plan = s.plan()
            self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, env in plan], ["0", "1"])
            self.assertTrue(all(env.get("STARTLUX_PREFIX_BIDIR") == "1" for _, env in plan))
            with mock.patch.dict("os.environ", {}, clear=False):
                self.assertNotIn("STARTLUX_PREFIX_BIDIR", VLLMServers(auto, ["0"], python="py").plan()[0][1])
            with mock.patch("startlux_decide.launch.plugin_visible", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "does not see"):
                    s.start()
        finally:
            shutil.rmtree(inst)
            shutil.rmtree(auto)


def mtp_folder(kind: str = "auto_think", layers: int = 1, tensors: bool = True) -> str:
    """A folder whose config declares MTP layers (as the released ones do) and whose index lists mtp.* tensors."""
    path = make_folder(kind, config={"text_config": {"mtp_num_hidden_layers": layers}})
    names = {"model.language_model.layers.0.mlp.up_proj.weight": "model-00001.safetensors"}
    if tensors:
        names.update({"mtp.fc.weight": "mtp-graft.safetensors", "mtp.norm.weight": "mtp-graft.safetensors"})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": names}, f)
    return path


def value(cmd: list, flag: str):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class VersionFlagsTest(unittest.TestCase):
    def setUp(self):
        self.folders = []

    def tearDown(self):
        for f in self.folders:
            shutil.rmtree(f, ignore_errors=True)

    def folder(self, *a, **kw) -> str:
        self.folders.append(mtp_folder(*a, **kw))
        return self.folders[-1]

    @staticmethod
    def command(folder: str, version: str | None, **kw) -> tuple:
        s = VLLMServers(folder, ["0"], python="py", vllm_version=version or "", **kw)
        return s.plan()[0][0], s

    def test_vllm_0191_unchanged(self):
        f = self.folder()
        for version in ("0.19.1", None, "0.30.2"):                     # old or unknown: the release command
            cmd, s = self.command(f, version)
            self.assertEqual(cmd, server_command("py", f, 8001), version)
            for switch in RELEASE_SWITCHES:
                self.assertIn(switch, cmd)
            self.assertEqual(value(cmd, "--max-num-batched-tokens"), "65536")
            self.assertNotIn("--speculative-config", cmd)
            self.assertIn("prefix caching and chunked prefill off", s.settings())

    def test_vllm_031_autothink_with_mtp(self):
        cmd, s = self.command(self.folder(), "0.31.0")
        for switch in ("--enable-prefix-caching", "--enable-chunked-prefill"):
            self.assertIn(switch, cmd)
        for switch in RELEASE_SWITCHES:
            self.assertNotIn(switch, cmd)
        self.assertEqual(value(cmd, "--max-num-batched-tokens"), "8192")
        self.assertEqual(json.loads(value(cmd, "--speculative-config")), {"method": "mtp", "num_speculative_tokens": 3})
        for flag, v in RELEASE_VALUES:                                  # everything else as in the release runs
            if flag != "--max-num-batched-tokens":
                self.assertEqual(value(cmd, flag), v, flag)
        self.assertEqual(s.mtp_tokens, 3)
        self.assertIn("MTP speculative decoding with 3 draft tokens", s.settings())

    def test_vllm_031_without_mtp(self):
        cmd, s = self.command(self.folder(), "0.31.0", mtp=False)       # switched off
        self.assertIn("--enable-prefix-caching", cmd)
        self.assertNotIn("--speculative-config", cmd)
        self.assertIn("switched off", s.settings())
        for folder in (self.folder(tensors=False), self.folder(layers=0)):     # no MTP weights / no MTP layers
            cmd, s = self.command(folder, "0.31.0")
            self.assertIn("--enable-prefix-caching", cmd)
            self.assertNotIn("--speculative-config", cmd)
            self.assertIn("holds no MTP weights", s.settings())

    def test_instant_folders(self):
        f = self.folder("instant")
        self.assertEqual(self.command(f, "0.19.1")[0], server_command("py", f, 8001))      # 0.19.1: unchanged
        cmd, s = self.command(f, "0.31.0")
        for switch in RELEASE_SWITCHES:                                 # whole prompts, no prefix cache, no MTP
            self.assertIn(switch, cmd)
        self.assertNotIn("--speculative-config", cmd)
        self.assertEqual(value(cmd, "--max-num-batched-tokens"), value(cmd, "--max-model-len"))
        self.assertEqual(s.plan()[0][1]["STARTLUX_PREFIX_BIDIR"], "1")

    def test_mtp_weights(self):
        self.assertTrue(has_mtp_weights(self.folder()))
        self.assertFalse(has_mtp_weights(self.folder(tensors=False)))
        self.assertFalse(has_mtp_weights(self.folder(layers=0)))
        single = make_folder(config={"mtp_num_hidden_layers": 1})       # one safetensors file, top-level config key
        self.folders.append(single)
        header = json.dumps({"mtp.fc.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
        with open(os.path.join(single, "model.safetensors"), "wb") as f:
            f.write(len(header).to_bytes(8, "little") + header + bytes(8))
        self.assertTrue(has_mtp_weights(single))
        self.assertFalse(has_mtp_weights("/nonexistent/folder"))


class StopTest(unittest.TestCase):
    def test_stop_ends_the_whole_process_group(self):
        """A stand-in server that starts a child and ignores SIGTERM: stop() must leave no process behind."""
        code = ("import signal, subprocess, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "subprocess.Popen([sys.executable, '-c', 'import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']); time.sleep(60)")
        s = VLLMServers("unused", ["0"])
        s.procs.append(subprocess.Popen([sys.executable, "-c", code], start_new_session=True))
        pgid = s.procs[0].pid
        time.sleep(0.5)
        t0 = time.time()
        s.stop(grace=1.0)
        self.assertLess(time.time() - t0, 10)
        time.sleep(0.2)
        self.assertFalse(_signal_group(pgid, 0))


if __name__ == "__main__":
    unittest.main()
