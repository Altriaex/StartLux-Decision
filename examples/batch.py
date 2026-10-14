"""Send many requests concurrently and write one answer line per request.

    python examples/batch.py requests.jsonl answers.jsonl [--url http://localhost:8090] [--workers 32]

Each input line is one request body ({"state": ..., "questions": ...}). Every worker opens its own connection per
request, so no connection is shared between threads."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from client import decide

ap = argparse.ArgumentParser()
ap.add_argument("requests")
ap.add_argument("answers")
ap.add_argument("--url", default="http://localhost:8090")
ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
bodies = [json.loads(line) for line in open(a.requests) if line.strip()]


def one(body):
    for attempt in range(3):
        try:
            return decide(a.url, body)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return {"error": err}


with ThreadPoolExecutor(a.workers) as pool, open(a.answers, "w") as out:
    for result in pool.map(one, bodies):
        out.write(json.dumps(result) + "\n")
print(f"{len(bodies)} requests answered into {a.answers}")
