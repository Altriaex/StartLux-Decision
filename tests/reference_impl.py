"""Reference copies used as test oracles: the prompt renderer the models were trained with, the loop detector of the
release runs (version 3), and the control flow of the proxy that produced the published AutoThink numbers (its
per-question pipeline, its read-out and its answer format). The code is copied as it was, with only the imports and
the HTTP call replaced (complete(url, body) goes to a fake vLLM). Do not "fix" anything here: the point is to compare
the package with what ran.
"""
import asyncio
import functools
import hashlib
import itertools
import json
import math
import re
import string

# ---------------------------------------------------------------- renderer (training format)
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 26
MAX_LEVELS = 10
_BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)
ANNOTATE_MIN = 8


def annotate_indices(x, min_len=ANNOTATE_MIN):
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def state_text(state):
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def value_text(v):
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _norm(s):
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def validate(row):
    t = row.get("type")
    if t not in TYPES:
        raise ValueError(f"unknown type {t!r}")
    if not isinstance(row.get("state"), str):
        raise ValueError("state must be a string")
    if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
        raise ValueError("instructions required")
    opts = row.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        raise ValueError(f"2..{MAX_OPTIONS} options required, got {len(opts) if isinstance(opts, list) else opts!r}")
    ids = []
    for o in opts:
        i = o.get("id")
        if not isinstance(i, str) or not i.strip() or "\n" in i or "\r" in i or len(i) > 200:
            raise ValueError(f"bad option id {i!r}")
        c = o.get("criterion")
        if c is not None and not isinstance(c, str):
            raise ValueError("criterion must be a string or None")
        ids.append(i)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate option id")
    if t == "noul" and set(ids) != {"true", "false"}:
        raise ValueError("noul ids must be true/false")
    if t == "score" and not 2 <= len(ids) <= MAX_LEVELS:
        raise ValueError(f"score needs 2..{MAX_LEVELS} levels")
    return ids


def option_lines(row, order):
    crit = {o["id"]: o.get("criterion") for o in row["options"]}
    t = row["type"]
    hide = t == "choice" and all(_BARE.match(i) for i in order) and all(crit[i] for i in order)
    lines = []
    for k, i in enumerate(order):
        name = ("yes" if i == "true" else "no") if t == "noul" else i
        c = crit[i]
        if hide:
            body = c
        elif c is None or not c.strip() or _norm(c) == _norm(name):
            body = name
        else:
            body = f"{name}: {c}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def messages(row, order=None):
    validate(row)
    order = [o["id"] for o in row["options"]] if order is None else list(order)
    if sorted(order) != sorted(o["id"] for o in row["options"]):
        raise ValueError("order must be a permutation of option ids")
    state = row["state"] if row["state"].strip() else "(none)"
    content = ("Evidence:\n" + state + "\n\nQuestion: " + row["instructions"].strip() + "\nOptions:\n"
               + "\n".join(option_lines(row, order)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], order


THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"


def check_tokenizer(tokenizer):
    probe, _ = messages({"type": "noul", "state": "s", "instructions": "q",
                         "options": [{"id": "true", "criterion": None}, {"id": "false", "criterion": None}]})
    text = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("tokenizer chat template lacks the thinking-off assistant prefix")
    ids = tokenizer.encode(text, add_special_tokens=False)
    letters = []
    for L in LETTERS:
        t = tokenizer.encode(L, add_special_tokens=False)
        if len(t) != 1 or tokenizer.encode(text + L, add_special_tokens=False) != ids + t:
            raise ValueError("letter is not a separate single token after the prefix: " + L)
        letters.append(t[0])
    if len(set(letters)) != 26:
        raise ValueError("letter ids not distinct")
    return letters


def score_keys(crit):
    try:
        return sorted(crit, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(crit)


def from_systemone(state, spec, qid="q"):
    t = spec.get("type", "choice")
    t = "noul" if t == "bool" else t
    crit = spec.get("criteria", spec.get("options"))
    ins = value_text(spec.get("instructions", spec.get("question"))) or ""
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        opts = [{"id": str(k), "criterion": value_text(v)} for k, v in crit.items()]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in score_keys(crit)]
        opts = [{"id": str(i), "criterion": value_text(c)} for i, c in enumerate(crit)]
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        tr, fa = c.get("true", c.get(True)), c.get("false", c.get(False))
        opts = [{"id": "true", "criterion": value_text(tr)}, {"id": "false", "criterion": value_text(fa)}]
        if not ins:
            ins = "Which answer fits the evidence?"
    else:
        raise ValueError(f"unknown question type {t!r}")
    row = {"id": qid, "type": t, "state": state_text(state), "instructions": ins, "options": opts}
    if t == "score":
        row["ordered"] = True
    return row


# ---------------------------------------------------------------- loop detector (versions 1-3)
_DIG, _WS = re.compile(r"\d+"), re.compile(r"\s+")


def _periodic(t, lo, hi, min_rep, min_span):
    n = len(t)
    for p in range(lo, hi + 1):
        r = max(min_rep, -(-min_span // p))
        if n >= r * p and all(t[n - (j + 1) * p:n - j * p] == t[n - p:] for j in range(1, r)):
            return True
    return False


def few_lines(text, lines=20, max_distinct=8):
    ls = [x for x in (_WS.sub(" ", y).strip() for y in text[-80 * lines:].split("\n")) if x]
    return len(ls) > lines and len(set(ls[-lines:])) <= max_distinct


def low_diversity(text, words=60, max_distinct=6):
    w = _DIG.sub("#", text[-12 * words:]).split()
    return len(w) > words and len(set(w[-words:])) <= max_distinct


def loopdet_looping(text, window=400, max_unit=80, min_rep=4, min_span=60, long_window=1600, long_unit=400, long_rep=3,
                    digit_run=150, version=3):
    u = _DIG.sub("#", text[-window:])
    for t in (u, _WS.sub(" ", u)):
        if _periodic(t, 1, max_unit, min_rep, min_span):
            return True
    if version < 2:
        return False
    d = text[-digit_run:]
    if len(d) == digit_run and sum(c.isdigit() for c in d) >= 0.9 * digit_run:
        return True
    v = _WS.sub(" ", _DIG.sub("#", text[-long_window:]))
    if _periodic(v, max_unit + 1, long_unit, long_rep, 0):
        return True
    return version >= 3 and (few_lines(text) or low_diversity(text))


def loopdet_loop_cut(text, **kw):
    if not loopdet_looping(text, **kw):
        return None
    lo, hi = 0, len(text)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        if loopdet_looping(text[:mid], **kw):
            hi = mid
        else:
            lo = mid
    return hi


# the release proxy ran with --loop-version 3: functools.partial(loopdet.looping, version=3)
looping = functools.partial(loopdet_looping, version=3)
loop_cut = functools.partial(loopdet_loop_cut, version=3)

# ---------------------------------------------------------------- the release proxy's per-question pipeline
CUT = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"


def answer(row, p, question):
    ids = [o["id"] for o in row["options"]]
    if row["type"] == "noul":
        return {"type": "noul", "noul": p[ids.index("true")]}
    if row["type"] == "score":
        levels = question.get("criteria") or []
        levels = list(levels.values()) if isinstance(levels, dict) else list(levels)
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": max(p),
                "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                "probabilities": {str(i): v for i, v in enumerate(p)}}
    dist = dict(zip(ids, p))
    best = max(dist, key=dist.get)
    return {"type": "choice", "choice": best, "confidence": dist[best], "probabilities": dist}


class WaveGen:
    """The release proxy's wave batching, as it was."""

    def __init__(self, proxy, url, max_wave, gather_s):
        self.proxy, self.url, self.max_wave, self.gather_s = proxy, url, max_wave, gather_s
        self.q = asyncio.Queue()
        self.task = asyncio.get_event_loop().create_task(self.worker())

    async def submit(self, body):
        fut = asyncio.get_event_loop().create_future()
        await self.q.put((body, fut))
        return await fut

    async def worker(self):
        while True:
            batch = [await self.q.get()]
            await asyncio.sleep(self.gather_s)
            while len(batch) < self.max_wave and not self.q.empty():
                batch.append(self.q.get_nowait())
            res = await asyncio.gather(*(self.proxy.complete(self.url, b) for b, _ in batch), return_exceptions=True)
            for (_, fut), r in zip(batch, res):
                if isinstance(r, BaseException):
                    fut.set_exception(r)
                else:
                    fut.set_result(r)


class Proxy:
    """decide_one of the release proxy. `a` holds its command-line settings (mode, tau, mix, budget, prefix,
    tail_chars, rep_pen, stop_loops, loop_chunk, max_len, debug_probs, no_waves, wave_max, wave_gather); `temp` the
    folder's temperature_by_type, `tthink` the think temperatures; complete(url, body) answers like vLLM."""

    def __init__(self, a, tok, temp, tthink, urls, complete):
        self.a = a
        self.tok = tok
        self.letter_ids = check_tokenizer(self.tok)
        self.temp = temp
        self.tthink = tthink
        self.urls = urls
        self.backends = itertools.cycle(self.urls)
        self.waves = {}
        self.name = "m"
        self.prefix = a.prefix.replace("\\n", "\n")
        self.complete = complete

    @staticmethod
    def salted(body, salt):
        if salt == "call":
            body["cache_salt"] = "x"
        elif salt:
            body["cache_salt"] = salt
        return body

    async def read(self, url, text, k, typ, salt=None):
        body = self.salted({"model": self.name, "prompt": text, "max_tokens": 1, "temperature": 1.0, "logprobs": k,
                            "allowed_token_ids": list(self.letter_ids[:k]), "return_tokens_as_token_ids": True}, salt)
        out = await self.complete(url, body)
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        z = [top.get(f"token_id:{t}", -1e9) for t in self.letter_ids[:k]]
        t = self.temp.get(typ, 1.0)
        m = max(z)
        e = [math.exp((x - m) / t) for x in z]
        s = sum(e)
        return [x / s for x in e]

    def prompts(self, msgs):
        plain = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        think = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if not think.rstrip().endswith("<think>"):
            think += "<think>\n"
        return plain, think + self.prefix

    async def decide_one(self, state, q, mode=None, salt=None):
        mode = mode or self.a.mode
        url = next(self.backends)
        wide = None
        if q.get("type", "choice") == "choice" and isinstance(q.get("criteria"), dict) and len(q["criteria"]) > MAX_OPTIONS:
            keys = list(q["criteria"])
            n = -(-len(keys) // 25)
            size, extra = divmod(len(keys), n)
            groups, start = [], 0
            for g in range(n):
                end = start + size + (1 if g < extra else 0)
                groups.append(keys[start:end])
                start = end
            fp, fin = {}, []

            async def first(g):
                row = from_systemone(state, dict(q, criteria={c: q["criteria"][c] for c in g}))
                msgs, _ = messages(row)
                return g, await self.read(url, self.prompts(msgs)[0], len(g), "choice", salt)
            for g, p in await asyncio.gather(*(first(g) for g in groups)):
                fp.update(dict(zip(g, p)))
                fin += [c for c, _ in sorted(zip(g, p), key=lambda x: -x[1])[:3]]
            if len(fin) > MAX_OPTIONS:
                fin = sorted(fin, key=lambda c: -fp[c])[:MAX_OPTIONS]
            wide = (fp, fin)
            q = dict(q, criteria={c: q["criteria"][c] for c in fin})
        row = from_systemone(state, q)
        msgs, _ = messages(row)
        plain, think = self.prompts(msgs)
        k, typ = len(row["options"]), row["type"]
        n_in = len(self.tok(plain, add_special_tokens=False)["input_ids"])
        p0 = await self.read(url, plain, k, typ, salt)
        p, n_out = p0, 0
        room = self.a.max_len - n_in - 64
        thought = (mode == "think" or (mode == "hybrid" and max(p0) < self.a.tau)) and room > 32
        p1 = None
        if thought:
            seed = int(hashlib.sha1(think.encode()).hexdigest()[:8], 16)
            budget, gen_text, n_out, closed, rnd = min(self.a.budget, room), "", 0, False, 0
            while n_out < budget:
                step = min(self.a.loop_chunk, budget - n_out) if self.a.loop_chunk else budget
                gen = {"model": self.name, "prompt": think + gen_text, "max_tokens": step,
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20, "stop": ["</think>"], "seed": (seed + 7919 * rnd) % (2 ** 31)}
                if self.a.rep_pen != 1.0:
                    gen["repetition_penalty"] = self.a.rep_pen
                gbody = self.salted(gen, salt)
                if self.a.no_waves:
                    g = await self.complete(url, gbody)
                else:
                    if url not in self.waves:
                        self.waves[url] = WaveGen(self, url, self.a.wave_max, self.a.wave_gather)
                    g = await self.waves[url].submit(gbody)
                c = g["choices"][0]
                got = (g.get("usage") or {}).get("completion_tokens", 0)
                gen_text += c["text"]
                n_out += got
                if c.get("finish_reason") == "stop":
                    closed = True
                    break
                if not self.a.loop_chunk or got == 0:
                    break
                if looping(gen_text):
                    gen_text = gen_text[:loop_cut(gen_text)]
                    closed = True
                    break
                rnd += 1
            th = (self.prefix + gen_text).replace("<think>", "", 1).replace("</think>", "").strip()
            if self.a.tail_chars and len(th) > self.a.tail_chars:
                th = "[...]\n" + th[-self.a.tail_chars:]
            base = think[:len(think) - len(self.prefix)] if self.prefix else think
            text = base + th + ("\n</think>\n\n" if closed else CUT)
            p1 = await self.read(url, text, k, typ, salt)
            if typ in self.tthink:
                qq = [max(v, 1e-300) ** (self.temp.get(typ, 1.0) / self.tthink[typ]) for v in p1]
                p1 = [v / sum(qq) for v in qq]
            p = [(1 - self.a.mix) * x + self.a.mix * y for x, y in zip(p0, p1)] if self.a.mix > 0 else p1
        if wide:
            fp, fin = wide
            final = dict(zip([o["id"] for o in row["options"]], p))
            rest = [c for c in fp if c not in final]
            mass = sum(fp[c] for c in rest) or 1.0
            dist = {c: final[c] * (1 - 1e-3) for c in final}
            dist.update({c: 1e-3 * fp[c] / mass for c in rest})
            z = sum(dist.values())
            dist = {c: v / z for c, v in dist.items()}
            best = max(dist, key=dist.get)
            out = {"type": "choice", "choice": best, "confidence": dist[best], "probabilities": dist}
        else:
            out = answer(row, p, q)
        out["_debug"] = {"options": [o["id"] for o in row["options"]], "p0": p0, "p1": p1, "n_out": n_out}
        return out, n_in, n_out

    async def handle(self, state, questions, mode=None):
        res = await asyncio.gather(*(self.decide_one(state, q, mode, None) for q in questions.values()))
        answers = {k: r[0] for k, r in zip(questions, res)}
        usage = {"input_tokens": sum(r[1] for r in res), "output_tokens": sum(r[2] for r in res)}
        return {"answers": answers, "usage": usage, "model": self.name}
