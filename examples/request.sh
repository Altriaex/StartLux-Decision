#!/usr/bin/env bash
# One request with three typed questions. Usage: bash examples/request.sh [http://localhost:8090]
URL=${1:-http://localhost:8090}
AUTH=()
[ -n "${STARTLUX_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${STARTLUX_API_KEY}")
curl -s "$URL/v1/systemone" -H 'Content-Type: application/json' "${AUTH[@]}" -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  },
  "return_details": true
}'
echo
