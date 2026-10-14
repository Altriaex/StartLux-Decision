# Quantization

`startlux-decide quantize fp8|gguf|mlx` prints the steps below (or runs them with `--run`). It checks that the tools
are there and downloads nothing.

## FP8 in vLLM

No separate build: vLLM quantizes the weights to FP8 when it loads the folder.

```bash
startlux-decide launch-vllm --model /models/StartLux-Decision-4B --gpus 0 --quantization fp8
```

This adds `--quantization fp8` to every server. The KV cache stays BF16 (the Instant plugin refuses an FP8 KV cache).

## GGUF for llama.cpp

AutoThink folders, and Instant folders when the llama.cpp tree has the converter patch `patches/llama.cpp/0002-convert-qwen35-prompt-bidirectional.patch` (it marks the GGUF non-causal; the command checks for it). For images, convert the vision encoder once with `convert_hf_to_gguf.py MODEL --mmproj --outtype f16` and pass the file to llama-server with `--mmproj`. With a llama.cpp checkout that has a build of `llama-quantize`:

```bash
startlux-decide quantize gguf --model /models/StartLux-Decision-4B --out /models/4B-gguf --llama-cpp ~/llama.cpp --type Q8_0 --run
llama-server -m /models/4B-gguf/model-Q8_0.gguf -ngl 99 -c 65536 --port 8081
startlux-decide serve --model /models/4B-gguf --backend llamacpp --llamacpp-url http://127.0.0.1:8081
```

The first step runs `convert_hf_to_gguf.py --outtype bf16` with the current Python (which needs llama.cpp's conversion
requirements), then `llama-quantize` (unless `--type BF16`), and copies the tokenizer files, `config.json` and
`decision_config.json` next to the GGUF file: the package renders the prompts and reads the letters itself. llama.cpp
must be recent enough to know the model's architecture. With `--parallel N`, llama-server splits `-c` over N slots; the
package takes the per-slot context from `/props` for the room check of thoughts.

`quantize gguf` makes the text decoder only. Images need the model's vision encoder in a separate mmproj file, which the
released GGUF repositories carry as an F16 file (`mmproj-*-F16.gguf`); see [engines.md](engines.md#llamacpp).

## MLX

Instant and AutoThink folders, on Apple Silicon. The text model is quantized (affine, group size 64), the vision tower
stays BF16, so the folder still reads images:

```bash
startlux-decide quantize mlx --model /models/StartLux-Decision-4B --out /models/4B-mlx-8bit --bits 8 --run
startlux-decide serve --model /models/4B-mlx-8bit --backend mlx
```

## Instant folders

A GGUF build of an Instant (prompt-bidirectional) folder must be read with the bidirectional prompt it was trained
with: convert it with `patches/llama.cpp/0002-...` (the released files already carry the flag) and serve it as
described in [engines.md](engines.md#llamacpp). Read causally, the build answers as a different model. The MLX engine reads
Instant folders bidirectionally, so MLX builds of them are fine.

## Measure every build again

A quantized build is a different model. Decision models are more sensitive to this than chat models in two ways:

* the answer is the probability distribution itself, and its calibration (ECE, Brier) can move even where the chosen
  option does not;
* AutoThink thinks when the first read-out's top probability is below 0.9. A shift of a few hundredths in that
  probability changes which questions think, and a question that thinks in one build and not in the other can end
  with a different answer, latency and token count.

Compare every build with BF16 on the same requests:

```bash
startlux-decide eval --url http://127.0.0.1:8090 --data gold.jsonl --out bf16.jsonl     # BF16 server
startlux-decide eval --url http://127.0.0.1:8091 --data gold.jsonl --out fp8.jsonl      # the build
startlux-decide eval --compare bf16.jsonl fp8.jsonl
```

`--compare` reports the agreement of the chosen answers, both accuracies and their difference with a 95% bootstrap
interval over requests, the agreement of the gate's decisions (thought or not), and the largest change of any option's
probability per question.

| model | build | engine | agreement with BF16 | accuracy BF16 → build (95% CI of the difference) | thought decisions agree | max \|Δp\| (mean) |
|---|---|---|---|---|---|---|
| 4B Instant | FP8 | vLLM | TODO | TODO | n/a | TODO |
| 4B AutoThink | FP8 | vLLM | TODO | TODO | TODO | TODO |
| 4B AutoThink | GGUF Q8_0 | llama.cpp | TODO | TODO | TODO | TODO |
| 4B AutoThink | GGUF Q4_K_M | llama.cpp | TODO | TODO | TODO | TODO |
| 4B AutoThink | MLX 8-bit | mlx-vlm | TODO | TODO | TODO | TODO |
