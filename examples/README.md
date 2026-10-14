# Examples

All examples talk to a running `startlux-decide serve` and use the Python standard library only. Pass the server with
`--url` (default `http://localhost:8090`) and set `STARTLUX_API_KEY` if the server checks API keys.

| File | What it does |
|---|---|
| [request.sh](request.sh) | one request with a choice, a yes/no and a score question, with per-answer details |
| [client.py](client.py) | the same request from Python, printing each answer |
| [image_request.py](image_request.py) | a yes/no question about a photo |
| [batch.py](batch.py) | many requests from a JSONL file, sent concurrently |

```bash
python examples/client.py --url http://localhost:8090
python examples/image_request.py parcel.jpg
python examples/batch.py requests.jsonl answers.jsonl --workers 32
```
