"""A deterministic fake of the vLLM /v1/completions, /v1/models and /version endpoints, at the JSON level.

Read-outs (bodies with "logprobs") get log-softmax values over the allowed letters, derived from a hash of the prompt.
Thought chunks continue a token stream fixed by the thinking prompt (everything up to the first opened think block
of the assistant turn); the first chunk of a thought fixes the forced prefix, so later chunks can tell how much of
the stream was already consumed. Stream kinds: closed by "</think>", ended by the end-of-sequence token, short / line /
long / digit loops, endless prose, and stalls (zero tokens with finish "length", as vLLM at the context limit).

A generation body with "stream" gets the same tokens as an async iterator of vLLM's stream events: `burst` tokens per
event (several per step, as with speculative decoding), continuous usage statistics, the finish reason on the last
choice event, then a usage event. Streams closed before their end are counted in `aborted`.

Calls with images come to /v1/chat/completions as the client sends them: the fake checks the pass-through template,
rebuilds the prompt from the message parts (text, and the image placeholder for each image), answers as for that
prompt (the images' data URIs enter the read-out hash) and replies in the chat format. A completions body with
max_tokens 1, temperature 0 and no stop string is a prefill (kept in `prefills`). With `strict`, a prompt longer
than max_model_len is refused as vLLM refuses it (BackendRequestError, an HTTP 400).
"""
import hashlib
import json
import math
import random

from startlux_decide.backends.base import BackendRequestError
from startlux_decide.images import IMAGE_PLACEHOLDER, PASS_THROUGH_TEMPLATE

HEADER = "<|im_start|>assistant\n<think>\n"
EOS = object()
STALL = object()
KINDS = ("close", "eos", "loop_short", "loop_lines", "loop_long", "digits", "endless", "stall")
WORDS = ("evidence", "suggests", "option", "because", "the", "ticket", "refund", "however", "we", "check", "date",
         "2026-03-14", "amount", "42", "is", "not", "clear", "so", "consider", "B", "A", "C", "<think>", "then", "rule",
         "applies", "here", "first", "second", "third", "price", "order", "late", "early", "x=3", "y", "thus", "fits")


def _rng(*parts):
    return random.Random(hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8", "surrogatepass")).digest())


def _prose(r, n):
    out = []
    for _ in range(n):
        w = r.choice(WORDS) if r.random() < 0.7 else f"w{r.randint(0, 5000)}"
        out.append(w + ("\n" if r.random() < 0.08 else " "))
    return out


def stream(head, kinds=KINDS):
    """The token stream of a thought for the thinking prompt `head`."""
    r = _rng("stream", head)
    kind = r.choice(kinds)
    n = r.choice((0, 3, 40, 300, 900, 1500, 3000, 5000))
    toks = _prose(r, n)
    if kind == "close":
        return toks + ["</think>"] + _prose(r, 50)
    if kind == "eos":
        return toks + [EOS]
    if kind == "loop_short":
        unit = r.choice(("B. ", "The answer is B.\n", "A\n", "so the answer is C, "))
        return toks + [unit] * 6000
    if kind == "loop_lines":
        lines = [f"Option {r.choice('ABC')} fits {i}.\n" for i in range(r.randint(2, 6))]
        return toks + [r.choice(lines) for _ in range(6000)]
    if kind == "loop_long":
        unit = " ".join(r.choice(WORDS[:20]) for _ in range(30)) + ".\n"
        return toks + [unit] * 2000
    if kind == "digits":
        return toks + ["3141592653"[i % 10] for i in range(9000)]
    if kind == "endless":
        return toks + _prose(r, 9000)
    return toks + [STALL]


class FakeVLLM:
    """transport(url, body) for VLLMBackend, and complete(url, body) for the reference proxy. Every completion body
    is logged (canonical JSON). drop_rate: share of read-outs that leave one letter out (vLLM never does with
    processed log-probabilities; the clients must still default it to -1e9)."""

    def __init__(self, max_model_len=65536, kinds=KINDS, drop_rate=0.0, version="0.19.1", burst=1, strict=False):
        self.max_model_len = max_model_len
        self.strict = strict
        self.prefills = []
        self.chat = []
        self.kinds = kinds
        self.drop_rate = drop_rate
        self.version = version                  # None: the server has no /version route
        self.burst = burst
        self.aborted = 0
        self.log = []
        self.prefixes = {}
        self.streams = {}

    async def __call__(self, url, body):
        if body is None:
            if url.endswith("/version"):
                if self.version is None:
                    raise LookupError("404: no /version route")
                return {"version": self.version}
            assert url.endswith("/v1/models"), url
            return {"object": "list", "data": [{"id": "m", "object": "model", "max_model_len": self.max_model_len}]}
        if url.endswith("/v1/chat/completions"):
            return self.chat_call(body)
        assert url.endswith("/v1/completions"), url
        self.log.append(json.dumps(body, sort_keys=True))
        if self.strict and len(body["prompt"]) + body["max_tokens"] > self.max_model_len:
            raise BackendRequestError(f"This model's maximum context length is {self.max_model_len} tokens")
        if "logprobs" not in body and body["max_tokens"] == 1 and body.get("temperature") == 0.0 and "stop" not in body:
            self.prefills.append((body["prompt"], ()))
            return {"choices": [{"index": 0, "text": "A", "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 1}}
        if "logprobs" in body:
            return self.readout(body)
        if body.get("stream"):
            return self.stream_events(body)
        return self.generate(body)

    def chat_call(self, body):
        """A call with images (module docstring)."""
        assert body["chat_template"] == PASS_THROUGH_TEMPLATE and body["add_generation_prompt"] is False
        assert "mm_processor_kwargs" not in body                # vLLM 0.31 refuses them per request
        (message,) = body["messages"]
        prompt, uris = "", []
        for part in message["content"]:
            if part["type"] == "text":
                prompt += part["text"]
            else:
                assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/")
                prompt += IMAGE_PLACEHOLDER
                uris.append(part["image_url"]["url"])
        self.chat.append((prompt, uris, body))
        flat = {k: v for k, v in body.items() if k not in ("messages", "chat_template", "add_generation_prompt",
                                                           "top_logprobs")}
        flat["prompt"] = prompt
        if "logprobs" not in body and body["max_tokens"] == 1 and body.get("temperature") == 0.0:
            self.prefills.append((prompt, tuple(uris)))
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "A"},
                                 "finish_reason": "length"}], "usage": {"completion_tokens": 1}}
        if body.get("logprobs"):
            assert body["logprobs"] is True
            out = self.readout(dict(flat, logprobs=body["top_logprobs"]), salt=uris)
            top = out["choices"][0]["logprobs"]["top_logprobs"][0]
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length",
                                 "logprobs": {"content": [{"token": next(iter(top)), "logprob": 0.0, "top_logprobs": [
                                     {"token": t, "logprob": v} for t, v in top.items()]}]}}], "usage": out["usage"]}
        if body.get("stream"):
            return self._chat_events(self.stream_events(flat))
        out = self.generate(flat)
        c = out["choices"][0]
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": c["text"]},
                             "finish_reason": c["finish_reason"]}], "usage": out["usage"]}

    async def _chat_events(self, events):
        try:
            async for ev in events:
                ev = dict(ev, object="chat.completion.chunk")
                ev["choices"] = [{"index": c["index"], "delta": {"content": c["text"]},
                                  "finish_reason": c["finish_reason"]} for c in ev["choices"]]
                yield ev
        finally:
            await events.aclose()

    complete = __call__

    def readout(self, body, salt=()):
        assert body["max_tokens"] == 1 and body["temperature"] == 1.0 and body["return_tokens_as_token_ids"] is True
        ids = body["allowed_token_ids"]
        assert body["logprobs"] == len(ids)
        r = _rng("readout", body["prompt"], *salt)
        z = [r.uniform(-9.0, 3.0) if r.random() < 0.8 else r.uniform(-30.0, 9.0) for _ in ids]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        top = {f"token_id:{t}": x - lse for t, x in zip(ids, z)}
        if "probe" not in body["prompt"] and r.random() < self.drop_rate and len(ids) > 2:
            del top[f"token_id:{ids[r.randrange(len(ids))]}"]
        return {"choices": [{"index": 0, "text": f"token_id:{ids[0]}", "finish_reason": "length",
                             "logprobs": {"top_logprobs": [top]}}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 1}}

    def generate(self, body):
        out, n, finish = self.tokens(body)
        return {"choices": [{"index": 0, "text": "".join(out), "finish_reason": finish}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n}}

    async def stream_events(self, body):
        """vLLM's stream of the generation (module docstring); the "[DONE]" line is the HTTP layer's."""
        assert body["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
        out, n, finish = self.tokens(body)
        items = [(t, 1) for t in out] + ([("", 1)] if finish == "stop" else [])       # the stop token is counted
        pt, done, sent = len(body["prompt"]), False, 0

        def usage(k):
            return {"prompt_tokens": pt, "completion_tokens": k, "total_tokens": pt + k}
        try:
            groups = [items[i:i + self.burst] for i in range(0, len(items), self.burst)] or [[]]
            for j, g in enumerate(groups):
                sent += sum(c for _, c in g)
                last = j == len(groups) - 1
                if last:
                    done = True
                yield {"object": "text_completion", "choices": [{"index": 0, "text": "".join(t for t, _ in g),
                                                                 "finish_reason": finish if last else None}],
                       "usage": usage(sent)}
            yield {"object": "text_completion", "choices": [], "usage": usage(sent)}
        finally:
            if not done:
                self.aborted += 1

    def tokens(self, body):
        """(text tokens, completion tokens, finish reason) of a generation body."""
        assert body["stop"] == ["</think>"], body["stop"]
        prompt = body["prompt"]
        i = prompt.index(HEADER) + len(HEADER)
        head, rest = prompt[:i], prompt[i:]
        if head not in self.prefixes:
            self.prefixes[head] = rest                      # the first chunk: the forced prefix only
            self.streams[head] = stream(head, self.kinds)
        prefix, toks = self.prefixes[head], self.streams[head]
        assert rest.startswith(prefix), "a later chunk must continue the thought"
        done = rest[len(prefix):]
        pos, length = 0, 0
        while length < len(done):
            length += len(toks[pos])
            pos += 1
        assert length == len(done) and "".join(toks[:pos]) == done, "chunks must end at token boundaries"
        out, n, finish = [], 0, "length"
        for t in toks[pos:pos + body["max_tokens"]]:
            if t is STALL:
                break
            n += 1
            if t is EOS or t == "</think>":
                finish = "stop"
                break
            out.append(t)
        return out, n, finish
