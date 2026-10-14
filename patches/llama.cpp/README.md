# Patches for llama.cpp

Tested on llama.cpp `aa94f2086147` (tag b11549, 10 October 2026). Each applies with `git apply` in the llama.cpp tree;
0003 applies to a stock tree and on top of 0001.

* `0002-convert-qwen35-prompt-bidirectional.patch`: `convert_hf_to_gguf.py` writes `qwen35.attention.causal = false`
  into the GGUF of an Instant (prompt-bidirectional) folder, so llama.cpp attends over the whole prompt in the
  full-attention layers while the linear-attention layers stay causal, as in training. The released Instant GGUF files
  already carry this flag; a stock GGUF can get it with `--override-kv qwen35.attention.causal=bool:false`.
* `0001-server-bidirectional-prompts.patch`: llama-server evaluates a bidirectional prompt alone and in one piece (no
  batching with other slots, no context-checkpoint split, no reuse of a cached prefix), so any `--parallel` is safe.
  It adds the request field `prompt_bidirectional`, which the server echoes in `generation_settings`; the package's
  llama.cpp backend sends it and checks the echo (`STARTLUX_LLAMACPP_BIDIR=request`; the default `auto` uses this mode
  for a server that does not report patch 0003).

* `0003-server-mtmd-prompt-whole-on-non-causal-context.patch`: for images with an Instant model. On a non-causal context
  llama-server evaluates a prompt that holds images whole, in one micro-batch (stock b11549 decodes the text before an
  image, the image, and the text after it as three micro-batches, which cuts the bidirectional attention: the read-outs of
  the 4B Instant Q8_0 model then differ from BF16 by 0.022 on average over 32 image questions, against 0.0055 with the
  patch). A prompt longer than `-ub` is refused, never cut. `/props` reports `mtmd_prompt_whole_when_non_causal`; the
  package's backend takes that as a server for non-causal contexts and works in mode `server` (the default `auto`).
  Causal contexts and prompts without images are unchanged.

Without patch 0001, serve Instant files with a stock llama-server and `--parallel 1 --ctx-checkpoints 0 -ub 16384`
(the micro-batch must hold the longest prompt), and set `STARTLUX_LLAMACPP_BIDIR=server`. With its default settings
the stock server batches prompts and splits off their last tokens, which costs the 4B Instant model 7.7 points on
DecideBench; reading the prompt causally costs it 26 points. Images need `--mmproj` and the same flags. See
[docs/engines.md](../../docs/engines.md).
