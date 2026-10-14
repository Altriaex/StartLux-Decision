"""The parts of the vLLM plugin that run without vLLM: the prompt-end rule, the switches, the folder flag, the hooks per
vLLM version, the guard against unpatched model runners, and the plan of a step (V1 and V2 runners, with stand-ins for
vLLM's objects; these need numpy and torch)."""
from __future__ import annotations

import copy
import importlib.util
import os
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from startlux_decide import vllm_plugin as P
from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.prompt import from_systemone, messages, plain_prompt, think_prompt

from .folders import make_folder

HAVE_NP_TORCH = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None


class _PluginState:
    """Saves and restores the plugin's module state (STATUS, hooks, sys.meta_path) around a test."""

    def __enter__(self):
        self.saved = (copy.deepcopy(P.STATUS), dict(P._HOOKS), list(sys.meta_path))
        return self

    def __exit__(self, *exc):
        P.STATUS.clear()
        P.STATUS.update(self.saved[0])
        P._HOOKS.clear()
        P._HOOKS.update(self.saved[1])
        sys.meta_path[:] = self.saved[2]
        return False


class PromptEndTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.pe = P.PromptEnd(self.tok)
        self.msgs = messages(from_systemone("evidence", {"type": "noul", "instructions": "Late?"}))

    def test_one_pass_prompt_is_all_bidirectional(self):
        ids = self.tok.encode(plain_prompt(self.tok, self.msgs))
        self.assertEqual(self.pe(ids), (len(ids), "one_pass"))

    def test_thinking_prompt_ends_after_the_opened_think_block(self):
        base = think_prompt(self.tok, self.msgs)
        ids = self.tok.encode(base + "Let me work through this.\nA stray <think>\n in the thought</think>\n\nB")
        self.assertEqual(self.pe(ids), (len(self.tok.encode(base)), "think"))

    def test_unmatched(self):
        self.assertEqual(self.pe(self.tok.encode("no chat template here")), (None, "unmatched"))


class SwitchesTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(P.ENV, "STARTLUX_PREFIX_BIDIR")
        self.assertEqual(P.EXTRA_KEY, "startlux_prompt_end")
        self.assertTrue(P.LEGACY_ENV.endswith("_PREFIX_BIDIR") and P.LEGACY_ENV != P.ENV)
        self.assertTrue(P.LEGACY_EXTRA_KEY.endswith("_prompt_end") and P.LEGACY_EXTRA_KEY != P.EXTRA_KEY)

    def test_new_name_first_then_alias(self):
        with mock.patch.dict(os.environ, {P.LEGACY_ENV + "_CHECK": "1"}, clear=False):
            self.assertEqual(P.env("_CHECK"), "1")
        with mock.patch.dict(os.environ, {P.LEGACY_ENV: "0", P.ENV: "1"}, clear=False):
            self.assertEqual(P.env(), "1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(P.env())
            self.assertEqual(P.env("_UNMATCHED", "error"), "error")

    def test_plugin_does_nothing_without_the_switch(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(P, "install") as install:
            P.plugin()
            install.assert_not_called()
        with mock.patch.dict(os.environ, {P.ENV: "1"}, clear=True), mock.patch.object(P, "install") as install:
            P.plugin()
            install.assert_called_once()

    def test_folder_flag(self):
        inst, auto = make_folder("instant"), make_folder()
        try:
            from startlux_decide.modelcfg import read_prompt_attention
            self.assertEqual((read_prompt_attention(inst), read_prompt_attention(auto)), ("bidirectional", "causal"))
            self.assertEqual(read_prompt_attention("/nonexistent/folder"), "causal")
        finally:
            shutil.rmtree(inst)
            shutil.rmtree(auto)


class InstallTest(unittest.TestCase):
    """install() hooks the model runners of each validated vLLM version (the hooks are recorded, not run)."""

    def install(self, version: str, allow: bool = False):
        calls = {}
        environ = {k: v for k, v in os.environ.items() if "ALLOW_VERSION" not in k}
        if allow:
            environ[P.ENV + "_ALLOW_VERSION"] = "1"

        def hook(name, fn):
            calls[name] = fn.__name__
            P.STATUS["hooked"].append(name)

        with _PluginState():
            P.STATUS.update(installed=False, hooked=[], patched=[], error=None, vllm=None)
            with mock.patch.object(P, "_hook", side_effect=hook), mock.patch("importlib.metadata.version",
                                                                            return_value=version), \
                    mock.patch.dict(os.environ, environ, clear=True):
                ok = P.install()
            return ok, calls, P.STATUS["error"]

    def test_validated_versions(self):
        self.assertEqual(P.VLLM_VERSIONS, ("0.19.1", "0.31.0"))

    def test_0191_patches_the_v1_runner_only(self):
        ok, calls, _ = self.install("0.19.1")
        self.assertTrue(ok)
        self.assertEqual(calls, {P.RUNNER_V1: "_patch_runner", "vllm.v1.attention.backends.flash_attn": "_patch_flash_attn",
                                 "vllm.v1.worker.gpu_worker": "_patch_worker"})

    def test_031_patches_both_runners_and_the_warmup(self):
        ok, calls, _ = self.install("0.31.0")
        self.assertTrue(ok)
        self.assertEqual(calls[P.RUNNER_V2], "_patch_runner_v2")
        self.assertEqual(calls[P.WARMUP_V2], "_patch_warmup")
        self.assertEqual(calls[P.RUNNER_V1], "_patch_runner")
        self.assertEqual(calls["vllm.v1.worker.gpu_worker"], "_patch_worker")

    def test_other_version_only_guards(self):
        ok, calls, err = self.install("0.32.0")
        self.assertFalse(ok)
        self.assertEqual(calls, {P.RUNNER_V1: "_guard_runner", "vllm.v1.worker.gpu_worker": "_patch_worker"})
        self.assertIn("0.19.1 and 0.31.0", err)
        ok, calls, _ = self.install("0.32.0", allow=True)
        self.assertTrue(ok)
        self.assertIn(P.RUNNER_V2, calls)


class WorkerGuardTest(unittest.TestCase):
    """A bidirectional folder must get a model runner the plugin patched; any other runner fails at engine start."""

    def setUp(self):
        self.inst, self.auto = make_folder("instant"), make_folder()

    def tearDown(self):
        shutil.rmtree(self.inst)
        shutil.rmtree(self.auto)

    @staticmethod
    def worker(model, runner):
        return SimpleNamespace(model_config=SimpleNamespace(model=model), model_runner=runner)

    def test_unpatched_runner_is_refused_for_bidirectional_folders_only(self):
        class GPUModelRunner:
            pass
        with self.assertRaisesRegex(RuntimeError, "GPUModelRunner is not patched"):
            P._require_patched_runner(self.worker(self.inst, GPUModelRunner()))
        P._require_patched_runner(self.worker(self.auto, GPUModelRunner()))
        patched = GPUModelRunner()
        patched._startlux_prefix = SimpleNamespace(bi=True)
        P._require_patched_runner(self.worker(self.inst, patched))

    def test_patched_worker_checks_after_init_device(self):
        folder = self.inst

        class Worker:
            def __init__(self):
                self.model_config = SimpleNamespace(model=folder)

            def init_device(self):
                self.model_runner = object()

        with _PluginState():
            P._patch_worker(SimpleNamespace(Worker=Worker))
            self.assertTrue(hasattr(Worker, "startlux_prefix_status"))
            with self.assertRaisesRegex(RuntimeError, "reads its prompt bidirectionally"):
                Worker().init_device()


class WarmupTest(unittest.TestCase):
    """vLLM 0.31 warms its kernels up with synthetic requests through the real path: no plan while that runs."""

    def test_window_sets_and_restores_the_flag(self):
        seen = []

        def warmup_kernels(model_runner, execute_model, sample_tokens):
            seen.append(model_runner._startlux_warmup)
            self.assertIsNone(P._make_plan_v2(model_runner, None))     # no plan (and no look at the batch)
            raise ValueError("boom")

        runner = SimpleNamespace()
        mod = SimpleNamespace(warmup_kernels=warmup_kernels, other=len)
        with _PluginState():
            P._patch_warmup(mod)
            self.assertTrue(mod.warmup_kernels._startlux_wrapped)
            self.assertIs(P._warmup_window(mod.warmup_kernels), mod.warmup_kernels)     # wrapped once
            before = P.STATUS["warmup_steps"]
            with self.assertRaises(ValueError):
                mod.warmup_kernels(runner, None, None)
            self.assertEqual(P.STATUS["warmup_steps"], before + 1)
        self.assertEqual(seen, [True])
        self.assertFalse(runner._startlux_warmup)
        self.assertIs(mod.other, len)

    def test_worker_binding_is_wrapped_only_with_the_v2_runner(self):
        def warmup_kernels(model_runner, *a):
            return model_runner._startlux_warmup

        class Worker:
            def init_device(self):
                pass
        with _PluginState():
            mod = SimpleNamespace(Worker=Worker, warmup_kernels=warmup_kernels)
            P._patch_worker(mod)
            self.assertIs(mod.warmup_kernels, warmup_kernels)          # 0.19.1: untouched
            P._HOOKS[P.RUNNER_V2] = P._patch_runner_v2
            P._patch_worker(mod)
            self.assertTrue(mod.warmup_kernels(SimpleNamespace()))


@unittest.skipUnless(HAVE_NP_TORCH, "needs numpy and torch")
class PlanTest(unittest.TestCase):
    """The plan of one step from V1 and from V2 runner state: decode r0 (1 token), prefill r1 (7 prompt tokens,
    p = 5), prefill r2 (6 prompt tokens, p = 6). p is passed explicitly, so no tokenizer is needed."""

    def setUp(self):
        self.state = _PluginState().__enter__()
        self.st = SimpleNamespace(device="cpu", logged=3, prompt_end=None)

    def tearDown(self):
        self.state.__exit__(None, None, None)

    @staticmethod
    def request(rid, n, p):
        return SimpleNamespace(req_id=rid, prompt_token_ids=list(range(n)), num_prompt=n,
                               sampling_params=SimpleNamespace(extra_args={P.EXTRA_KEY: p}))

    def v1_runner(self, sched, computed):
        import numpy as np
        reqs = {"r0": self.request("r0", 8, 8), "r1": self.request("r1", 7, 5), "r2": self.request("r2", 6, 6)}
        ib = SimpleNamespace(req_ids=["r0", "r1", "r2"], num_computed_tokens_cpu=np.array(computed),
                             num_prompt_tokens=np.array([8, 7, 6]))
        return SimpleNamespace(input_batch=ib, requests=reqs, _startlux_prefix=self.st), dict(zip(ib.req_ids, sched))

    def v2_runner(self, sched, computed):
        import numpy as np
        reqs = {"r0": self.request("r0", 8, 8), "r1": self.request("r1", 7, 5), "r2": self.request("r2", 6, 6)}
        qs = np.concatenate([[0], np.cumsum(sched)]).astype(np.int32)
        ib = SimpleNamespace(num_reqs=3, req_ids=["r0", "r1", "r2"], num_computed_tokens_np=np.array(computed),
                             is_prefilling_np=np.array(computed) < np.array([8, 7, 6]),
                             num_scheduled_tokens=np.array(sched, dtype=np.int32), query_start_loc_np=qs,
                             num_tokens=int(qs[-1]), has_prefill=True)
        return SimpleNamespace(_startlux_reqs=reqs, _startlux_prefix=self.st), ib

    def test_v1_and_v2_give_the_same_plan(self):
        r1, sched = self.v1_runner([1, 7, 6], [9, 0, 0])
        a = P._make_plan(r1, 3, sched)
        r2, ib = self.v2_runner([1, 7, 6], [9, 0, 0])
        b = P._make_plan_v2(r2, ib)
        for plan in (a, b):
            self.assertEqual(plan.idx.tolist(), [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
            self.assertEqual(plan.cu.tolist(), [0, 5, 11])
            self.assertEqual((plan.maxp, plan.ntok, plan.rows), (6, 14, 11))
            self.assertEqual(plan.reqs, [(1, 1, 7, 5), (2, 8, 6, 6)])

    def test_no_prefill_no_plan(self):
        r1, sched = self.v1_runner([1, 1, 1], [9, 8, 7])
        self.assertIsNone(P._make_plan(r1, 3, sched))
        r2, ib = self.v2_runner([1, 1, 1], [9, 8, 7])
        self.assertIsNone(P._make_plan_v2(r2, ib))

    def test_a_split_prompt_raises(self):
        r2, ib = self.v2_runner([1, 3, 6], [9, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "split across steps"):
            P._make_plan_v2(r2, ib)

    def test_resumed_prefill_is_counted(self):
        r2, ib = self.v2_runner([1, 9, 6], [9, 0, 0])          # r1 re-prefills 7 prompt + 2 generated tokens
        before = P.STATUS["resumed_prefills"]
        plan = P._make_plan_v2(r2, ib)
        self.assertEqual(P.STATUS["resumed_prefills"], before + 1)
        self.assertEqual(plan.reqs, [(1, 1, 9, 5), (2, 10, 6, 6)])

    def test_unknown_request_raises(self):
        r2, ib = self.v2_runner([1, 7, 6], [9, 0, 0])
        del r2._startlux_reqs["r2"]
        with self.assertRaisesRegex(RuntimeError, "was not seen as a new request"):
            P._make_plan_v2(r2, ib)


@unittest.skipUnless(HAVE_NP_TORCH, "needs numpy and torch")
class RunnerV2Test(unittest.TestCase):
    """_patch_runner_v2 on a stand-in for vLLM 0.31's GPUModelRunner: requests are recorded and dropped, the plan
    reaches the FLASH_ATTN metadata of a prefill step, FULL CUDA graphs and a replaced model state are refused."""

    def setUp(self):
        self.state = _PluginState().__enter__()
        P.STATUS["installed"] = True
        self.folder = make_folder("instant")

    def tearDown(self):
        shutil.rmtree(self.folder)
        self.state.__exit__(None, None, None)

    def make_runner(self):
        folder = self.folder

        class FlashAttentionMetadata:
            def __init__(self, n):
                self.num_actual_tokens = n

        class GDNMeta:
            pass
        GDNMeta.__module__ = "vllm.v1.attention.backends.gdn_attn"

        class ModelState:
            def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
                return {"layers.3.attn": FlashAttentionMetadata(input_batch.num_tokens), "layers.0.attn": GDNMeta()}

        class GPUModelRunner:
            def __init__(self, vllm_config, device):
                self.model_config = SimpleNamespace(model=folder, tokenizer=folder, hf_config=None, max_model_len=64,
                                                    is_encoder_decoder=False)
                self.scheduler_config = SimpleNamespace(enable_chunked_prefill=False, long_prefill_token_threshold=0,
                                                        max_num_batched_tokens=64)
                self.cache_config = SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto")
                self.parallel_config = SimpleNamespace()
                self.speculative_config = None
                self.device = device
                self.removed = []

            def load_model(self, *args, **kwargs):
                self.model_state = ModelState()

            def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
                pass

            def add_requests(self, scheduler_output):
                pass

            def _remove_request(self, req_id):
                self.removed.append(req_id)
                return True

            def execute_model(self, *args, **kwargs):
                return "ran"

        mod = SimpleNamespace(GPUModelRunner=GPUModelRunner)
        P._patch_runner_v2(mod)
        runner = GPUModelRunner(None, "cpu")
        runner.load_model()
        runner._startlux_prefix.logged = 3
        return runner, ModelState

    def test_prefill_step(self):
        import numpy as np
        runner, _ = self.make_runner()
        self.assertEqual(runner._startlux_prefix.kind, "V2")
        new = SimpleNamespace(req_id="a", prompt_token_ids=list(range(5)), prompt_len=5,
                              sampling_params=SimpleNamespace(extra_args={P.EXTRA_KEY: 5}))
        runner.add_requests(SimpleNamespace(scheduled_new_reqs=[new]))
        self.assertIn("a", runner._startlux_reqs)
        ib = SimpleNamespace(num_reqs=1, req_ids=["a"], num_computed_tokens_np=np.array([0]),
                             is_prefilling_np=np.array([True]), num_scheduled_tokens=np.array([5], dtype=np.int32),
                             query_start_loc_np=np.array([0, 5], dtype=np.int32), num_tokens=5, has_prefill=True)
        md = runner.model_state.prepare_attn(ib, SimpleNamespace(name="PIECEWISE"))
        plan = md["layers.3.attn"].startlux_plan
        self.assertEqual((plan.idx.tolist(), plan.cu.tolist()), ([0, 1, 2, 3, 4], [0, 5]))
        self.assertFalse(hasattr(md["layers.0.attn"], "startlux_plan"))
        with self.assertRaisesRegex(RuntimeError, "FULL CUDA graph"):
            runner.model_state.prepare_attn(ib, SimpleNamespace(name="FULL"))
        decode = SimpleNamespace(**{**vars(ib), "has_prefill": False})
        self.assertFalse(hasattr(runner.model_state.prepare_attn(decode, None)["layers.3.attn"], "startlux_plan"))
        self.assertEqual(runner.execute_model(), "ran")
        runner._remove_request("a")
        self.assertEqual((runner._startlux_reqs, runner.removed), ({}, ["a"]))

    def test_replaced_model_state_is_refused(self):
        runner, model_state_cls = self.make_runner()
        runner.model_state = model_state_cls()
        with self.assertRaisesRegex(RuntimeError, "was replaced after load_model"):
            runner.execute_model()


@unittest.skipUnless(HAVE_NP_TORCH, "needs numpy and torch")
class FixupMemoryTest(unittest.TestCase):
    """The fix-up of a long step (views of the rows, written into the output in place) gives the same rows as the
    gathered fix-up of short steps; a fake varlen attention stands in for FlashAttention."""

    def test_lean_equals_gathered(self):
        import sys
        import types

        import numpy as np
        import torch

        calls = []

        def varlen(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, softmax_scale, causal, out=None,
                   **kw):
            calls.append(out is not None)
            assert not causal
            res = torch.empty_like(q) if out is None else out
            cu = cu_seqlens_q.tolist()
            rep = q.shape[1] // k.shape[1]
            for a, b in zip(cu[:-1], cu[1:]):
                kk, vv = k[a:b].repeat_interleave(rep, 1), v[a:b].repeat_interleave(rep, 1)
                w = torch.softmax(torch.einsum("qhd,khd->hqk", q[a:b].float(), kk.float()) * softmax_scale, -1)
                res[a:b] = torch.einsum("hqk,khd->qhd", w, vv.float()).to(res.dtype)
            return res
        fake = types.ModuleType("fa_utils")
        fake.flash_attn_varlen_func = varlen
        names = ["vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends",
                 "vllm.v1.attention.backends.fa_utils"]
        saved = {n: sys.modules.get(n) for n in names}
        for n in names[:-1]:
            sys.modules.setdefault(n, types.ModuleType(n)) if saved[n] is None else None
        sys.modules[names[-1]] = fake
        try:
            g = torch.Generator().manual_seed(0)
            ntok, heads, kv, dim = 60, 4, 2, 8
            q, k, v = (torch.randn(ntok, h, dim, generator=g) for h in (heads, kv, kv))
            impl = types.SimpleNamespace(alibi_slopes=None, sinks=None, kv_cache_dtype="auto", scale=0.3,
                                         sliding_window=(-1, -1), logits_soft_cap=0.0, vllm_flash_attn_version=2,
                                         batch_invariant_enabled=False)

            def plan_of():
                plan = P.Plan()
                reqs = [(0, 3, 20, 20), (1, 30, 25, 17)]                  # rows [3, 23) and [30, 47)
                rows = [np.arange(a0, a0 + p) for _, a0, _, p in reqs]
                plan.idx = torch.from_numpy(np.concatenate(rows))
                plan.cu = torch.tensor([0, 20, 37], dtype=torch.int32)
                plan.maxp, plan.reqs, plan.ntok, plan.rows, plan.single = 20, reqs, ntok, 37, None
                return plan
            outs = {}
            for lean in ("1", "1000000"):
                base = torch.randn(ntok, heads, dim, generator=torch.Generator().manual_seed(1))
                with mock.patch.dict(os.environ, {"STARTLUX_PREFIX_BIDIR_LEAN_ROWS": lean}):
                    calls.clear()
                    P._fixup(impl, plan_of(), q, k, v, base)
                outs[lean] = (base, list(calls))
            self.assertEqual(outs["1"][1], [True, True])                 # one call per request, into the output
            self.assertEqual(outs["1000000"][1], [False])                 # one gathered call
            self.assertTrue(torch.equal(outs["1"][0], outs["1000000"][0]))
            untouched = torch.randn(ntok, heads, dim, generator=torch.Generator().manual_seed(1))
            keep = [i for i in range(ntok) if not (3 <= i < 23 or 30 <= i < 47)]
            self.assertTrue(torch.equal(outs["1"][0][keep], untouched[keep]))     # decode rows left as they were
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m


if __name__ == "__main__":
    unittest.main()
