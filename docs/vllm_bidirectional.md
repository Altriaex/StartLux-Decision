# Prompt-bidirectional attention in vLLM

Instant folders were trained to read their prompt with bidirectional attention in the full-attention layers. Stock
vLLM reads every prompt causally, so it would run such a folder as a different model, without any error. The plugin
in `startlux_decide/vllm_plugin.py` makes vLLM (V1 engine) read these folders the way they were trained. It is
validated on vLLM 0.19.1 and 0.31.0 (see [vLLM versions](#vllm-versions)). Nothing in the installed vLLM is edited: the
plugin patches classes when vLLM imports them.

## The mask

In the full-attention layers (one layer in four), inside one request:

    query q may read key k  <=>  k <= q  or  (q < p and k < p)

The Gated DeltaNet layers, their short convolutions and the KV-cache write stay exactly as in stock vLLM.

The prompt end p of a request comes from its prompt token ids:

| prompt | p |
|---|---|
| ends with the empty think block `<think>\n\n</think>\n\n` (one-pass read-out) | L: the whole input is bidirectional |
| otherwise | right after the `<think>\n` that follows the first `<\|im_start\|>assistant\n` (system turn, evidence, question, options, assistant header). The forced prefix, the thought, `</think>` and the read-out after it are causal |
| `SamplingParams(extra_args={"startlux_prompt_end": p})`, or `"vllm_xargs": {"startlux_prompt_end": p}` in an OpenAI request | that p (0 = causal, L = all bidirectional) |
| none of the above | an error (or `STARTLUX_PREFIX_BIDIR_UNMATCHED=causal` / `full`) |

On 570 test prompts per model the rule gave the same p as the transformers reference. A stray `<think>` inside a
thought does not move p.

## How it works

1. For each step, a hook where the model runner builds the attention metadata finds the requests whose prompt is
   prefilled in this step: `GPUModelRunner._build_attention_metadata` in vLLM's model runner V1, the model state's
   `prepare_attn` in model runner V2 (the default of 0.31). It checks that each such prompt starts at position 0 and is
   scheduled whole, and raises otherwise. It attaches a small plan to the FLASH_ATTN metadata: the batch rows
   `[start, start + p)` and the prompt lengths.
2. `FlashAttentionImpl.forward` first runs the stock causal FlashAttention call. Then, for the planned rows only, it
   runs one non-causal varlen FlashAttention call (the kernel vLLM already uses) over each request's own prompt q/k/v.
   The k/v are the tensors the stock code wrote to the KV cache. The result overwrites those rows.
3. Rows at or after p keep the stock causal result, which is exactly the mask for q >= p.
4. Later steps need nothing. The prompt K/V in the cache were computed from the bidirectional prompt, and every later
   query is later than every prompt key. Decode steps are untouched: full CUDA graphs for decode batches are as in
   stock vLLM. A step with a prefill runs attention eagerly between the piecewise graphs, and the fix-up runs there.
5. A request that is preempted is prefilled again from position 0 with its generated tokens, in one step. The same
   plan applies.

The extra work is the causal pass over the prompt rows, which is thrown away: +2.2% wall time in our measurement.

## Enable

Install this package in the Python environment that runs vLLM (it declares the vLLM general plugin
`startlux_prefix_bidir`) and set `STARTLUX_PREFIX_BIDIR=1`. Without the variable the plugin does nothing at all.
`startlux-decide launch-vllm` does both and checks the result. Engine settings for a bidirectional folder (the plugin
checks them at engine start and refuses to run otherwise):

```bash
vllm serve DIR --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 65536 --max-model-len 65536 ...
```

`--max-num-batched-tokens` must be at least `--max-model-len`, so that each prompt is prefilled whole in one step. The
attention backend must be FLASH_ATTN (vLLM's default here).

vLLM calls general plugins in the engine-core process and in every worker before the worker is built. The plugin only
registers import hooks, so non-vLLM processes and the API server are not touched. If `VLLM_PLUGINS` is set, it must list
`startlux_prefix_bidir`. In-process alternative: `VLLM_ENABLE_V1_MULTIPROCESSING=0`, then
`from startlux_decide import vllm_plugin; vllm_plugin.install()`.

**Check that an engine has it.** The server log prints `prefix-bidirectional prompt attention ON for DIR` and the p of
the first three requests; `launch-vllm` refuses to continue without that line. For an in-process `vllm.LLM`,
`vllm_plugin.require_active(llm, DIR)` raises unless every worker reports the plugin active
(`llm.collective_rpc("startlux_prefix_status")`; stock vLLM has no such method).

**When it is off.** With `STARTLUX_PREFIX_BIDIR` unset nothing is patched. With it set and a causal folder, the patched
methods call the stock code only. In both cases the outputs were bit-identical to stock vLLM (test d below).

**Debug.** `STARTLUX_PREFIX_BIDIR_CHECK=1` recomputes the attention of up to `STARTLUX_PREFIX_BIDIR_CHECK_REQS` (2)
requests per layer call in fp32 with the explicit masks. It raises if the fix-up is further from its reference than
twice the stock causal kernel is from its own (or one bf16 ulp at the largest output).

The variable names and the extra-argument key of the plugin's earlier internal copy are accepted as aliases.

## vLLM versions

| vLLM | model runner | where the plugin finds the prompts of a step |
|---|---|---|
| 0.19.1 (the release runs) | V1, `vllm.v1.worker.gpu_model_runner` | `GPUModelRunner._build_attention_metadata`, from the persistent input batch and the cached request states |
| 0.31.0 | V2, `vllm.v1.worker.gpu.model_runner` (the default) | the model state's `prepare_attn`, from the step's `InputBatch`; prompt ids and sampling parameters from the new requests of the scheduler output |
| 0.31.0 with `VLLM_USE_V2_MODEL_RUNNER=0` | V1 | as on 0.19.1 |

The fix-up itself (`FlashAttentionImpl.forward`, one non-causal varlen FlashAttention call) is the same code on both
versions. Both are validated ([results](#vllm-0310)). Other versions are refused unless
`STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1`.

**Why 0.31 needed a port.** vLLM 0.31 builds model runner V2 by default (its log says "Using V2 Model Runner"). The
plugin for 0.19.1, run there with the version check waived, patched only the V1 class: it logged "installed" but never
"prefix-bidirectional prompt attention ON", and vLLM read the Instant 4B causally without an error. Against 0.19.1 with
the plugin it chose the same option on 95.1% of 823 questions and moved 23.7% of the read-outs by more than 0.05.

**A runner the plugin does not know is refused.** After a worker has built its model runner, the plugin checks that a
bidirectional folder got a runner it patched; otherwise the engine stops at start with the runner's class name. This
also refuses the experimental runner V2 of 0.19.1 (`VLLM_USE_V2_MODEL_RUNNER=1`).

**Kernel warm-up on 0.31.** Before it serves, vLLM 0.31 warms its kernels up with synthetic requests sent through the
real model path (prompt ids 0, 1, 2, ...). They match no prompt-end rule. While the warm-up runs the plugin makes no
plan, so these requests are read causally; their outputs are thrown away. The status counter `warmup_steps` counts
such steps.

**Engine settings on 0.31.** The flags are those of 0.19.1. In 0.31 prefix caching on these hybrid models works only
together with chunked prefill; the plugin needs both off, as before. A step with a prompt prefill must not replay a
FULL CUDA graph (the fix-up runs in eager attention); with the default `FULL_AND_PIECEWISE` such steps run piecewise,
and the plugin raises if one is ever dispatched to a FULL graph. TRITON_ATTN is refused on 0.31 even with
`STARTLUX_PREFIX_BIDIR_ALLOW_TRITON=1`: 0.31 turns its PrefixLM ranges into a tensor when the metadata is built, so
the plan cannot reach them.

## Test results

### vLLM 0.19.1

Measured with the plugin before it moved into this package (same logic; only names changed), bf16, one H200 per run.
Models: a prompt-bidirectional thinking test model (B), a one-pass Instant 4B candidate (I), and for the noise floor a
causal 4B (C). The transformers reference is the training code's bidirectional forward. Token ids were rendered once,
so both sides read identical ids. Δp is the largest change of an option's probability per item.

**Noise floor.** The same comparison for the causal model C, stock vLLM against transformers: one-pass mean |Δp| 0.0061,
max 0.040, top-1 agreement 99.0%; read-outs after a thought: mean 0.0055, top-1 99.0%; greedy 128 tokens: 50% identical.

| test | vLLM with the plugin vs transformers reference | stock vLLM vs transformers reference |
|---|---|---|
| (a) one-pass read-out, model B, 300 items | mean \|Δp\| 0.0065, max 0.048, top-1 99.3% | 0.097, max 0.74, top-1 88.3% |
| (a) one-pass read-out, model I, 300 items | mean 0.0053, max 0.031, top-1 99.7% | 0.097, max 0.75, top-1 88.3% |
| (c) read-out after a stored thought, 200 thoughts | mean 0.0060, max 0.052, top-1 99.5% | 0.067, max 0.57, top-1 95.0% |
| (b) greedy thought, 128 tokens, 50 prompts | 62% identical; first difference after 94 tokens on average; first-token top-1 98%, mean \|Δlogprob\| 0.014 | 2% identical; 22 tokens |

For (b): the transformers reference against itself, with prompts decoded one at a time instead of batched, gives 74%
identical and a first-token mean |Δlogprob| of 0.014, so the vLLM-vs-reference difference is batching noise.

| more checks | result |
|---|---|
| (d) plugin off, bit-identical | Stock vs plugin loaded without the variable (model B): identical. Stock vs plugin active on a causal folder (model C): identical. Each over 570 cases (12,315 floats and every token), twice, with a schedule that does not depend on timing (in-process engine, no async scheduling; with async scheduling stock vLLM is not reproducible against itself) |
| detected p vs explicit p | bit-identical offline; the same over HTTP up to batching noise |
| fp32 check (570 cases, 4 requests per layer call, 1,000 checks) | fix-up error at most 2.03 × the stock causal kernel's own error on the same rows (max 0.148 vs 0.170 at outputs up to about 37: bf16 rounding). The bidirectional part changes the prompt rows by at least 4.4 |
| mask probe (prompt log-probabilities) | changing the last user-turn token moves the log-probabilities of earlier positions by 0.67 to 1.86 with the plugin, by 0 without. Changing a thought token moves nothing before it (exactly 0) and at least 2.4 after it |
| preemption (small KV cache, 6 requests prefilled again) | all 248 checked fix-ups, the re-prefills included, pass the fp32 check (ratio ≤ 1.21) |
| `vllm serve` with the plugin | one-pass mean \|Δp\| 0.0066, top-1 99.3%; read-out after a thought 0.0059, 99.5%; serve vs offline identical up to batching noise |
| held-out run, 3,021 items (model B, sampled thoughts, budget 4,096) | one-pass read-out: mean \|Δp\| 0.0069, top-1 99.0%; accuracy 72.49% (reference) vs 72.46% (vLLM). After the thought: 74.81% vs 74.94%, difference +0.13 points, 95% CI −0.56 to +0.86. Brier 0.359 vs 0.357. Thoughts closed 17.7% vs 18.1%, 3,538 vs 3,539 tokens. GPU time 4.3 × lower on vLLM |
| throughput, one H200 (thought generation, budget 4,096, temperature 0.6 / top-p 0.95 / top-k 20, read-outs before and after) | 12,728 thoughts per hour including read-outs (15,744 generation only), 9,920 tokens/s; the transformers reference runner about 3,500 thoughts per hour: 3.6 × (4.5 × generation only) |
| overhead (1,024 × 512 forced tokens, same engine settings) | 54.6 s with the plugin, 53.4 s stock (+2.2%) |

### vLLM 0.31.0

Measured on 11 October 2026 with this package, bf16, one H200 per engine, FlashAttention 3 on both versions. Instant
release folders (4B unless noted). Offline tests run `vllm.LLM` in one process without async scheduling, on 300
one-pass prompts from the latency requests (questions with 2 to 26 options), 40 prompts read after a short written
thought, 20 one-pass prompts with an explicit p, and greedy continuations of 32 tokens. Δp is the largest change of an
option's probability per question.

| test | result |
|---|---|
| 0.19.1, the plugin before the port against this one, same GPU one after the other | bit-identical: all 360 read-outs, all 40 continuations, the same plan counters. Two runs of either plugin are also bit-identical. (The very first runs, two engines compiling their kernels at the same time on empty caches, differed at batching-noise level: 146 of 300 one-pass read-outs bit-identical, mean \|Δp\| 0.0014, same top option on all; with filled caches every repeat was bit-identical) |
| fp32 check on 0.31, 4 requests per layer call | runner V2: 216 checks, fix-up error at most 1.52 × the stock kernel's own error on the same rows. Runner V1: 232 checks, at most 1.52 ×. The bidirectional part changes the prompt rows by at least 4.6 |
| preemption on 0.31 (36 KV blocks of 528 tokens, 16 sequences, continuations forced to 512 tokens), fp32 check on | 6 requests prefilled again with their generated tokens on each runner; all 976 checks pass (at most 1.75 × on V2, 1.49 × on V1) |
| explicit p on 0.31 | p = L gives the detected one-pass read-out bit for bit (10 of 10); p = 0 reads causally (mean \|Δp\| 0.035) |
| causal folder (AutoThink 4B) on 0.31, plugin switched on and off | bit-identical (240 read-outs, 20 continuations) |
| runner V2 against runner V1 on 0.31 | same top option on 340 of 340 read-outs, 329 bit-identical; continuations identical |
| 0.19.1 against 0.31 (runner V2), offline | same top option on 340 of 340; one-pass mean \|Δp\| 0.0015, max 0.025, none above 0.05; 38 of 40 continuations identical |
| unknown runner | 0.19.1 with `VLLM_USE_V2_MODEL_RUNNER=1`: the engine refuses the Instant folder at start |

Through the server: `startlux-decide serve` in front of `vllm serve` with the release flags, the 300 latency requests
(823 questions) one at a time, `think_mode` never. Reference: vLLM 0.19.1 with the plugin before the port.

| engine | Instant 4B: same option, mean \|Δp\|, share above 0.05 | Instant 27B |
|---|---|---|
| 0.31.0 + this plugin, runner V2 | 99.88% (one question), 0.0017, 0.12% | 100%, 0.0019, 0.12% |
| 0.31.0 + this plugin, runner V1 | 99.88%, 0.0017, 0.12% | 100%, 0.0019, 0.12% |
| 0.19.1 + this plugin | 100%, 0.00001, 0% | 100%, 0.00006, 0% |
| 0.31.0 + the plugin before the port (version check waived; read causally) | 95.1%, 0.050, 23.7% | not run |
| floor between the versions: causal 27B AutoThink, 0.31.0 against 0.19.1, same flags | | 100%, 0.0023, 0.24% |

Latency of the one-pass read-out: the 300 latency requests one at a time through the release client, pass 2,
weighted to the Decision Index mix of prompt lengths (short 90.2%, medium 9.0%, long 0.79%); median / mean / p80 in
ms per request; one H200 per server, the same node and folders for every column.

| size | 0.19.1 + plugin | 0.31.0 + plugin | 0.31.0 without the plugin (prompt read causally, timing only) |
|---|---|---|---|
| 0.8B | 31 / 45 / 46 | 33 / 46 / 44 | 33 / 45 / 45 |
| 2B | 31 / 48 / 47 | 31 / 46 / 45 | 32 / 46 / 44 |
| 4B | 36 / 60 / 55 (repeat 36 / 60 / 53) | 39 / 62 / 53 (repeat 43 / 64 / 56) | 38 / 59 / 51 |
| 9B | 34 / 68 / 53 (repeat 35 / 79 / 55) | 40 / 82 / 62 (repeat 40 / 73 / 57) | 39 / 70 / 57 |
| 27B | 60 / 158 / 120 | 62 / 164 / 120 | 62 / 160 / 119 |

vLLM 0.31 answers short requests 1 to 7 ms later than 0.19.1 and the longest ones 20 to 31 ms sooner (median of the
long band: 4B 541 to 514 ms, 27B 2,602 to 2,582 ms). The same holds without the plugin, so this is the engine. The
plugin's own cost on 0.31 is 0 to 4 ms on short requests and 4 to 9% on the longest prompts (the extra non-causal
attention over the prompt). A single short request that stalls for a second moves the weighted mean by about 9 ms
(seen on both versions), so compare medians and p80.

## Known limits

- **FLASH_ATTN only.** TRITON_ATTN through vLLM's own PrefixLM ranges was tried as a cross-check. Its read-outs match
  the reference, but in 0.19.1 `triton_unified_attention` with `USE_MM_PREFIX` widens every sequence of a launch to
  `max(causal bound, seq_len)` keys: decode rows that share a launch with such a prefill read past their end, and
  generations collapse into `!!!!`. The plugin refuses TRITON_ATTN unless `STARTLUX_PREFIX_BIDIR_ALLOW_TRITON=1`
  (read-out-only jobs, 0.19.1 only; on 0.31 it is always refused).
- **Validated on H200** (FlashAttention 3 on both versions). On Blackwell, vLLM 0.31 picks FlashAttention 4; run the
  fp32 check (`STARTLUX_PREFIX_BIDIR_CHECK=1`) before trusting it there.
- **Required engine settings.** Chunked prefill and prefix caching off, `max_num_batched_tokens >= max_model_len`. A
  long prefill stalls the decodes of its step.
- **Images** (vLLM 0.31). Image tokens are prompt tokens like any other: the fix-up recomputes their rows with the rest
  of the prompt. With 12 image requests, the read-outs of the 4B and 27B Instant folders had the same top option as
  the transformers reference in 12 and 11 of 12 questions, largest probability difference 0.037 and 0.072; the causal
  AutoThink folders, which do not use the plugin, differed as much (0.073 and 0.058), and without the images the same
  questions differed by at most 0.007 (4B Instant) and 0.006 (27B Instant). The image processors of the two engines
  give identical pixels, so the difference comes from the two implementations of the vision path, not from the
  plugin. With vLLM's profile of the image encoder (as many full-size images as one step holds, 256 at 262,144
  tokens) the 4B server did not start (an illegal memory access in the profile run); the launcher passes
  `--skip-mm-profiling` to Instant servers with images.
- **Long prompts.** A prompt is prefilled whole in one step, so `max_num_batched_tokens` must reach the longest prompt
  and vLLM reserves the activations of such a step. The fix-up of a step with 16,384 prompt rows or more works in
  place (no copies of queries, keys, values or outputs; the same probabilities as the gathered call on a 32K prompt),
  because vLLM's memory profile runs without it. On vLLM 0.31 and one H200, 128,000-token prompts ran at 4B and 27B
  (7.3 s and 24.7 s), but the step of a 196,000-token prompt (27B) and of a 256,000-token prompt (4B) crashed the
  server (an illegal memory access; the 4B also ran out of memory), and the 27B does not start at 262,144 tokens
  (vLLM's own profile of such a step runs out of memory). `launch-vllm` therefore gives Instant servers 131,072
  tokens on 0.31; causal folders read long prompts in chunks and reach 262,144.
- **Not supported:** speculative decoding, decode context parallelism, ubatching (DBO), FP8 KV cache, prompt
  embeddings; on vLLM 0.19.1 multimodal inputs (text only: `--limit-mm-per-prompt` 0).
- **Literal header in user text.** If the evidence contains `<|im_start|>assistant\n<think>\n`, the detection of a
  thinking prompt would end the prompt there. Pass p explicitly in such pipelines. One-pass prompts are not affected
  (their rule looks at the end of the prompt).
- **Empty thought.** A read-out after an empty thought without a forced prefix (`template + "" + "\n</think>\n\n"`) is
  the same text as the one-pass prompt, so it is read fully bidirectionally.
- **Plugin not loaded.** `STARTLUX_PREFIX_BIDIR=1` without the package in vLLM's environment gives stock vLLM, which
  reads the folder causally without notice. Use `launch-vllm`, `require_active` or check the log.
- **Version.** Tied to the internals of vLLM 0.19.1 and 0.31.0: `FlashAttentionImpl.forward`; for runner V1
  `GPUModelRunner._build_attention_metadata` and the cached request states; for runner V2 `GPUModelRunner.add_requests`,
  `_remove_request`, `load_model`, `execute_model`, the model state's `prepare_attn` and the kernel warm-up functions.
  Other versions are refused; `STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1` overrides after re-validating (run the fp32 check
  and compare read-outs with a validated version).
- **Status counters on runner V2.** A request resumed after preemption comes back to runner V2 as a new request, so
  its prompt end is found again and counted twice in `bi_requests` and the prompt-kind counters (diagnostics only).
- **Prompt log-probabilities.** For a bidirectional prompt these are not language-model probabilities: a position sees
  later prompt tokens.
