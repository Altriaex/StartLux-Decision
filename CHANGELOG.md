# Changelog

## 1.1.0 (unreleased)

First public release of the engine, for the StartLux-Decision v1.1 models (Instant and AutoThink) and the earlier v1
models.

- Decision server with the `/v1/systemone` request format (also served as `/v1/decisions`), typed questions
  (`choice`, `noul`, `score`) and a probability for every option.
- AutoThink path: confidence gate, thought generation with a loop check, second read-out and mixture, with request
  controls `think_mode`, `gate`, `think_budget` and `mix`.
- `allow_none`: an opt-in "None of the above" option for choice questions, with `none_probability` in the answer.
- Default gate 0.6, so that more requests are answered in one pass. The published numbers used 0.9 (`--gate 0.9`).
- Engines: vLLM 0.19.1 and 0.31 (with a plugin for prompt-bidirectional Instant models), transformers, llama.cpp
  (GGUF, with patches for Instant models) and MLX (mlx-vlm).
- Images in the evidence and prompts up to 262,144 tokens.
- Production serving: API keys, TLS, early 429 responses, an access log and an optional request log.
- Tools: `launch-vllm`, `bench`, `eval` and `quantize` (FP8, GGUF and MLX builds).
