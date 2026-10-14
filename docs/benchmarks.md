# Benchmarks

## Accuracy and calibration: `startlux-decide eval`

Input: JSONL, one request per line, with gold labels:

```json
{"id": "r1", "state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {...}}},
 "gold": {"team": "billing"}}
```

The label is an option id for a choice, `true` / `false` (or `"yes"` / `"no"`) for a yes/no question, and the level
index (0 = lowest) or a criteria key for a score. Questions without a label are sent but not scored.

```bash
startlux-decide eval --url http://127.0.0.1:8090 --data gold.jsonl --out run.jsonl --concurrency 16
startlux-decide eval --score run.jsonl                    # score a predictions file again
startlux-decide eval --compare run_a.jsonl run_b.jsonl    # paired comparison
startlux-decide eval ... --controls '{"think_mode": "never"}'   # merged into every request
```

Every request is sent with `return_details`. Per question, the predicted label is the choice, yes when P(yes) ≥ 0.5, or
the most probable level; its probability is the confidence.

| metric | definition |
|---|---|
| accuracy | share of scored questions whose predicted label is the gold label |
| ECE | top-label calibration error: 10 equal-width confidence bins, \|accuracy − mean confidence\| per bin weighted by its share |
| Brier | sum over the options of (p − 1[option is gold])², averaged (a yes/no question counts both options) |
| thought share | share of questions that thought (from the answers' details) |
| thought tokens | mean thought tokens over all questions, and over the questions that thought |

`--compare A B` pairs questions by request id and key (same gold label) and reports the agreement of the predicted
labels, both accuracies, the difference B − A with a 95% bootstrap interval (2,000 resamples of requests, since the
questions of a request share their evidence), the both / only-A / only-B correct counts, the agreement of the gate's
decisions, and the largest change of any option's probability per question (mean and maximum). Use it to compare
engines, builds (BF16 against FP8, GGUF or MLX) and settings.

## Latency: `startlux-decide bench`

```bash
startlux-decide bench --url http://127.0.0.1:8090 --requests requests.jsonl --concurrency 1 --warmup 20 \
    --out latency_summary.csv --rows latency_rows.csv
```

The requests file has one `/v1/systemone` body per line (`id` and `gold` fields are ignored). Every request is sent
with `return_details`. Per request:

| column | measured by | meaning |
|---|---|---|
| `client_ms` | the client | from sending the request to receiving the whole answer |
| `wall_ms` | the server | the request inside the server, from validation to the answers |
| `first_pass_sum_ms` | the server | sum over the questions of their first-pass engine calls |
| `thought_span_ms` | the server | from the first thought chunk sent to the last one received, over all questions (0 without a thought) |
| `after_thought_sum_ms` | the server | sum over the questions of the read-outs after a thought |
| `questions`, `thought_questions`, `thought_tokens` | the server | per request |

The server's timings wrap the engine calls themselves: prompt rendering and tokenization are outside the per-phase
times and inside `wall_ms`. Thought spans include the wait for a wave (up to `--wave-gather`, 0.3 s, per chunk) on
vLLM. The summary CSV has n, mean, median, p80 and p95 of every column over the requests that succeeded (percentiles
by linear interpolation between order statistics).

Latency depends on the engine, the GPU, the share of questions that think and the load. For numbers comparable with the
published ones measure one request at a time (`--concurrency 1`) after a warm-up, on servers with the release flags and
nothing else running on their GPUs, and report the share of requests with a thought next to the latency.

## Reproducing the published numbers

Serve the folder on vLLM with the defaults (`startlux-decide serve --model DIR --launch-vllm --gpus ...`) and send the
benchmark's requests. The defaults are the settings of the release runs (README, "Published numbers"); for AutoThink
the mixture 0.75 is the one the published numbers use (it was applied to the stored first and second read-outs of every
question). Thoughts are sampled, so a rerun of an AutoThink benchmark differs from the published run within sampling
noise; Instant read-outs are deterministic up to the numerics of batching.
