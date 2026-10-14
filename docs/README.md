# Documentation

| Document | Read it when you want to |
|---|---|
| [api.md](api.md) | write a client: request and response format, controls, confidence, `none_probability`, errors |
| [none_probability.md](none_probability.md) | act on `none_probability`: what it means, how it was measured, why the threshold is 0.2 |
| [deployment.md](deployment.md) | run a server: engine settings to switch on or off, images and long inputs, v1 models |
| [engines.md](engines.md) | understand how each engine is driven and what was measured |
| [vllm_bidirectional.md](vllm_bidirectional.md) | understand the vLLM plugin that serves prompt-bidirectional Instant models |
| [quantization.md](quantization.md) | build or choose FP8, GGUF and MLX versions of a model |
| [serving_publicly.md](serving_publicly.md) | put the server on the internet: keys, TLS, rate limits, logs |
| [benchmarks.md](benchmarks.md) | reproduce accuracy and latency numbers |
| [../patches/llama.cpp](../patches/llama.cpp) | patch llama.cpp for Instant GGUF builds |
