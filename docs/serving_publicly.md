# Serving publicly

`startlux-decide serve` can face the internet directly: API keys, HTTPS, early 429s, the OpenRouter provider routes,
an access log without content and a request log with content. All of these are off by default.

| option | effect |
|---|---|
| `--host 0.0.0.0 --port 8443` | listen on every interface (the default is 127.0.0.1, local only) |
| `--api-keys-file PATH` | every route except `/health` needs `Authorization: Bearer <key>` with a key of the file, else 401 |
| `--ssl-certfile PATH --ssl-keyfile PATH` | HTTPS with TLS 1.2 or newer; plain HTTP when not given |
| `--max-inflight N` | when N decision requests are in flight, further ones get 429 at once with `Retry-After: 1`; nothing is queued |
| `--openrouter-models-file PATH` | the OpenRouter provider models document, served as it is at `--openrouter-models-path` (default `/openrouter/models`) |
| `--access-log PATH` | one JSON line per request, without content |
| `--request-log DIR` | one JSON line per decision request with its content: client address, key id, body, answers and thoughts; images as files |

`POST /v1/decisions` is the same API as `POST /v1/systemone` (OpenRouter's name for it). A `model` field in the request
is accepted and ignored; the response's `model` is the server's `--name`.

## Example

The 27B AutoThink model on four GPUs, one vLLM server per GPU (bound to 127.0.0.1), the decision server on port 8443:

```bash
startlux-decide serve --model /models/StartLux-Decision-27B-AutoThink --name startlux-decision-27b-autothink \
    --launch-vllm --gpus 0,1,2,3 --vllm-log-dir /var/log/startlux/vllm \
    --host 0.0.0.0 --port 8443 \
    --ssl-certfile /etc/startlux/tls/fullchain.pem --ssl-keyfile /etc/startlux/tls/privkey.pem \
    --api-keys-file /etc/startlux/api_keys --max-inflight 64 \
    --openrouter-models-file /etc/startlux/openrouter_models.json \
    --access-log /var/log/startlux/access.jsonl
```

Everything not listed keeps the defaults (vLLM flags, gate 0.6, budget 4,096, mixture 0.75). If a vLLM server
does not start for lack of GPU memory, raise `--gpu-memory-utilization` (0.88) when the GPUs run nothing else, or lower
`--max-model-len` (the launcher sets `--max-num-batched-tokens` to the same value; prompts longer than the context then
get a 422).

## API keys

* One key per line; blank lines and lines starting with `#` are ignored. A key is one word of printable ASCII, at
  least 16 characters. Make keys with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`, one per client.
* The file is read again when it changes (modification time, size or inode), checked on every request. Replace it
  atomically (write a new file, then rename it over the old one). A version with an invalid line is not applied: the
  server keeps the keys it had and logs the line number. A valid file without keys locks every client out. At start the
  file must exist and hold at least one key.
* Keys are kept only as SHA-256 digests and compared in constant time with every key. They never appear in logs or
  error messages. The logs name a key by its key id, the first 16 hex digits of its SHA-256 (for a key file line:
  `printf %s "$line" | sha256sum | cut -c1-16`). Keep the file readable by the server's user only (`chmod 600`).
* `/health` and `/v1/health` stay open for supervisors and load balancers; they return no request data. Without a
  valid key they return only the status, model name and kind, not the engine addresses or settings.

## TLS

The certificate must be valid for the public host name and come from a public CA (for example Let's Encrypt). The
certificate is loaded at start: restart the server after renewing it, or put a TLS-terminating proxy in front and serve
plain HTTP on 127.0.0.1 behind it (then keep `--api-keys-file` on the decision server).

## Early 429

OpenRouter counts 429 responses separately from errors and asks providers to answer them early instead of queueing.
With `--max-inflight N` the server counts the decision requests being answered; at N it answers 429 immediately, with
`Retry-After: 1` and a JSON error, before reading the request body. Unauthenticated requests get 401 first and never
count. Choose N with `startlux-decide bench --concurrency` on the real hardware: the concurrency at which p95 latency is
still acceptable, keeping in mind that requests that think take seconds. Declare the same number as capacity in the
OpenRouter document.

## OpenRouter models document

OpenRouter reads the provider's models from a document the provider serves. Its schema is not public for the
decisions modality, so the operator writes the file and the server serves it unchanged (it checks only that the file
is JSON with a non-empty `data` array of objects with a string `id`; the file is read again when it changes, and a
broken version is not applied). [openrouter_models.example.json](openrouter_models.example.json) uses the fields of
OpenRouter's provider documentation:

| field | example | note |
|---|---|---|
| `id` | `startlux/startlux-decision-27b-autothink` | the model's OpenRouter slug |
| `name`, `description` | | shown to users |
| `created` | `1791676800` | Unix time |
| `schema_version` | `"2.4"` | |
| `input_modalities` | `[{"type": "text"}]` | |
| `output_modalities` | `[{"type": "decisions"}]` | the decisions modality: OpenRouter must confirm this value |
| `quantization` | `"bf16"` | |
| `datacenters` | `[{"country_code": "CN"}]` | where the GPUs are |
| `is_free` | `true` | |
| `capacity` | `[{"type": "rate", "unit": "requests", "per": "minute", "value": 600}, {"type": "concurrency", "value": 32}]` | optional; confirm the entry types with OpenRouter, and keep the concurrency at or below `--max-inflight` |

## Access log

`--access-log` appends one JSON line per request:

```json
{"time":"2026-10-11T08:15:02.317Z","request_id":"4f1c...","method":"POST","route":"/v1/decisions","status":200,
 "total_ms":2210.5,"client_ip":"203.0.113.7","key_id":"9c1e5a0b2f64d317","questions":3,"input_tokens":912,
 "thought_tokens":640,"wall_ms":2209.8,"first_pass_sum_ms":98.1,"thought_span_ms":2061.4,"after_thought_sum_ms":44.0}
```

`route` is the matched route (null for unknown paths; query strings are never written). `client_ip` is the first
`X-Forwarded-For` entry, else `X-Real-IP`, else the TCP peer. `key_id` names the key of the request (null without a
bearer key; also given for keys that were refused). Token counts and timings are null for requests that did not reach
the model (401, 429, 4xx before the questions were read). The file is kept open: rotate it with `copytruncate`
(logrotate) or restart the server. The server's own log (stderr) has no request content either: errors carry the
request id, the exception type and the stack, and engine failures their engine message. A 502 tells the client only
the request id; a 500 the request id and the exception type.

## Request log

`--request-log DIR` keeps what clients sent and what they got back, for operators who must keep it (for example to
investigate complaints or abuse). Every decision request (`POST /v1/decisions`, `POST /v1/systemone`), whatever its
status, adds one JSON line to `DIR/requests-YYYY-MM-DD.jsonl` (a new file every UTC day):

| field | content |
|---|---|
| `time`, `request_id`, `route`, `status`, `total_ms` | as in the access log |
| `client` | `ip` (as `client_ip` above), `peer` (the TCP peer), `x_forwarded_for` and `x_real_ip` as received, `user_agent` |
| `key_id` | the key of the request (see API keys); the key itself is never written |
| `body_bytes` | the body size the client declared |
| `request` | the request body as parsed; each image replaced by `{"sha256", "mime", "bytes", "file"}`; `{"unparsed": ...}` (first 65,536 characters) for a body that is not JSON; null when the request was refused before its body was read (401, 429) |
| `response` | the reply; for answered requests with every answer's details and its thought text, also when the request did not ask for them (the client still gets only what it asked for) |

Images are written once each to `DIR/images/<first 2 hex digits>/<sha256>.<ext>`. The directory is created with mode
700 and every file with mode 600. A failed write is logged once and never fails the request. Nothing is deleted: plan
the retention (for example a daily `find DIR -name 'requests-*.jsonl' -mtime +30 -delete` and the same for images) and
state it in the privacy policy, because these files hold your users' data. OpenRouter shows each provider's data
policy to its users, so declare the retention there too.

## Checklist

- [ ] `--host 0.0.0.0` (or the public interface) and a port the firewall opens; no other port open. The vLLM servers
      listen on 127.0.0.1 and have no authentication: never expose them.
- [ ] `--api-keys-file` with one key per client, file mode 600, replaced atomically when keys change.
- [ ] `--ssl-certfile` / `--ssl-keyfile` with a publicly trusted certificate for the host name; a restart after
      renewal.
- [ ] `--max-inflight` from a bench run on the same GPUs; the same value (or lower) as capacity in the OpenRouter
      document.
- [ ] `--openrouter-models-file` written for this deployment (id, name, created, description, country code, capacity)
      and the output modality confirmed with OpenRouter.
- [ ] `--access-log` on, with rotation. `--request-log` only when content must be kept, with a retention job and the
      data policy saying so. vLLM with the launcher's flags writes no prompts to its logs; do not add request logging
      to it.
- [ ] A process supervisor that restarts the server and stops it with SIGTERM (the server then stops the vLLM servers it
      launched), with enough open files for the in-flight requests (two sockets each). `serve` raises its own soft
      limit to 65,536 at start, within the hard limit, and warns when the hard limit keeps it below 8,192.
- [ ] `/health` monitored from outside; the GPU validation (`startlux-decide eval` on a known set) passed on these GPUs.

A systemd unit:

```ini
[Unit]
Description=StartLux-Decision decision server
After=network-online.target
Wants=network-online.target

[Service]
User=startlux
ExecStart=/opt/startlux/venv/bin/startlux-decide serve --model /models/StartLux-Decision-27B-AutoThink ...
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=120
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

Request bodies are limited to 64 MB. Requests are not limited in their number of questions; keys are the control
over who can send them.
