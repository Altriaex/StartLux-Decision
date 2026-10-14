"""Ask three typed questions about one support ticket and print the answers.

    python examples/client.py [--url http://localhost:8090]

Set STARTLUX_API_KEY when the server was started with --api-keys-file. Standard library only."""
import argparse
import json
import os
import urllib.request

REQUEST = {
    "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]},
    },
}


def decide(url, body, timeout=300):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STARTLUX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STARTLUX_API_KEY"]
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8090")
    a = ap.parse_args()
    out = decide(a.url, REQUEST)
    for name, ans in out["answers"].items():
        if ans["type"] == "choice":
            print(f"{name}: {ans['choice']} (confidence {ans['confidence']:.2f})")
        elif ans["type"] == "noul":
            print(f"{name}: P(yes) = {ans['noul']:.2f}")
        else:
            print(f"{name}: expected level {ans['score']:.2f} of {len(ans['legend']) - 1}")
