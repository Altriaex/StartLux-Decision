"""Scoring and paired comparison of predictions files, and the latency summary."""
from __future__ import annotations

import unittest

from startlux_decide.bench import percentile, request_body, row_of, summarize
from startlux_decide.evaluate import compare, gold_label, prediction, record, score

CHOICE = {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None, "c": None}}
SCORE = {"type": "score", "instructions": "q", "criteria": {"2": "high", "0": "low", "1": "mid"}}


def rec(rid, answers, gold, status="ok"):
    """A predictions-file line with answers {key: answer} and gold {key: (type, label)}."""
    return {"id": rid, "status": status, "answers": answers,
            "questions": {k: {"type": t, "gold": g} for k, (t, g) in gold.items()}}


def choice(best, p, used=False, tokens=0):
    probs = {"a": 0.0, "b": 0.0, "c": 0.0}
    probs[best] = p
    for k in probs:
        if k != best:
            probs[k] = (1 - p) / 2
    return {"type": "choice", "choice": best, "probabilities": probs,
            "details": {"thought": {"used": used, "tokens": tokens}}}


class GoldTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(gold_label(CHOICE, "b"), "b")
        self.assertEqual(gold_label({"type": "choice", "criteria": [1, 2]}, 2), "2")
        self.assertEqual([gold_label({"type": "noul"}, g) for g in (True, "no", "YES", 0)], ["true", "false", "true",
                                                                                              "false"])
        self.assertEqual(gold_label(SCORE, "1"), "1")
        self.assertEqual(gold_label(SCORE, "2"), "2")
        self.assertEqual(gold_label({"type": "score", "criteria": {"lo": 1, "hi": 2}}, "hi"), "1")
        self.assertEqual(gold_label({"type": "score", "criteria": ["x", "y"]}, 1), "1")
        for spec, g in ((CHOICE, "d"), ({"type": "noul"}, "maybe"), (SCORE, 3), (SCORE, True)):
            with self.assertRaises(ValueError):
                gold_label(spec, g)
        with self.assertRaisesRegex(ValueError, "request 7, question q"):
            record({"id": 7, "questions": {"q": CHOICE}, "gold": {"q": "z"}}, 0)

    def test_predictions(self):
        self.assertEqual(prediction("noul", {"noul": 0.5}), ("true", {"true": 0.5, "false": 0.5}))
        self.assertEqual(prediction("score", {"probabilities": {"0": 0.4, "1": 0.4, "2": 0.2}})[0], "0")


class ScoreTest(unittest.TestCase):
    def test_metrics(self):
        records = [rec(1, {"q": choice("a", 0.9, True, 100)}, {"q": ("choice", "a")}),
                   rec(2, {"q": choice("b", 0.6)}, {"q": ("choice", "a")}),
                   rec(3, {"q": {"type": "noul", "noul": 0.8}}, {"q": ("noul", "true")}),
                   rec(4, {}, {"q": ("noul", "true")}, status="http_502"),
                   rec(5, {"q": choice("c", 0.5)}, {"q": ("choice", None)})]
        m = score(records)
        self.assertEqual((m["requests"], m["failed_requests"], m["questions"]), (5, 1, 3))
        self.assertAlmostEqual(m["accuracy"], 200 / 3)
        # bins: 0.9 -> correct, 0.6 -> wrong, 0.8 -> correct: |0.9-1| + |0.6-0| + |0.8-1| over 3
        self.assertAlmostEqual(m["ece"], 100 * (0.1 + 0.6 + 0.2) / 3)
        brier = ((0.9 - 1) ** 2 + 2 * 0.05 ** 2) + (0.6 ** 2 + (0.2 - 1) ** 2 + 0.2 ** 2) + (0.2 ** 2 + 0.2 ** 2)
        self.assertAlmostEqual(m["brier"], brier / 3)
        self.assertAlmostEqual(m["thought_share"], 50.0)                 # of the two answers with details
        self.assertAlmostEqual(m["thought_tokens_mean_when_thinking"], 100.0)
        self.assertEqual(m["by_type"]["noul"], {"questions": 1, "accuracy": 100.0})


class CompareTest(unittest.TestCase):
    def test_paired(self):
        a = [rec(i, {"q": choice("a", 0.7, i % 2 == 0)}, {"q": ("choice", "a" if i < 60 else "b")}) for i in range(100)]
        b = [rec(i, {"q": choice("a" if i < 80 else "b", 0.65, i % 3 == 0)}, {"q": ("choice", "a" if i < 60 else "b")})
             for i in range(100)]
        c = compare(a, b, resamples=500)
        self.assertEqual((c["paired_questions"], c["paired_requests"]), (100, 100))
        self.assertAlmostEqual(c["agreement"], 80.0)
        self.assertAlmostEqual(c["accuracy_a"], 60.0)
        self.assertAlmostEqual(c["accuracy_b"], 80.0)
        self.assertAlmostEqual(c["accuracy_diff_b_minus_a"], 20.0)
        lo, hi = c["accuracy_diff_ci95"]
        self.assertTrue(lo < 20.0 < hi and lo > 0)
        self.assertEqual((c["both_correct"], c["only_a_correct"], c["only_b_correct"]), (60, 0, 20))
        self.assertAlmostEqual(c["max_abs_dp"], 0.7 - 0.175)            # option a: 0.7 in A, 0.175 in B
        self.assertGreater(c["thought_agreement"], 0)
        self.assertEqual(compare(a, [], resamples=10), {"paired_questions": 0})


class BenchStatsTest(unittest.TestCase):
    def test_percentiles(self):
        v = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual([percentile(v, q) for q in (0, 50, 80, 95, 100)], [10.0, 30.0, 42.0, 48.0, 50.0])
        self.assertEqual(percentile([7.0], 95), 7.0)

    def test_rows_and_summary(self):
        resp = {"answers": {"x": {"details": {"thought": {"used": True, "tokens": 300}}},
                            "y": {"details": {"thought": {"used": False, "tokens": 0}}}},
                "timing_ms": {"wall": 120.0, "first_pass_sum": 30.0, "thought_span": 80.0, "after_thought_sum": 9.0}}
        row = row_of(0, {"id": "r1"}, "ok", 125.0, resp)
        self.assertEqual((row["id"], row["questions"], row["thought_questions"], row["thought_tokens"],
                          row["thought_span_ms"]), ("r1", 2, 1, 300, 80.0))
        rows = [row, row_of(1, {}, "http_502", 3.0, None, "bad gateway")]
        s = {r["metric"]: r for r in summarize(rows)}
        self.assertEqual((s["client_ms"]["n"], s["client_ms"]["median"], s["wall_ms"]["p95"]), (1, 125.0, 120.0))
        self.assertEqual(request_body({"id": 1, "gold": {}, "state": "s", "questions": {}}, {"gate": 0.5}),
                         {"state": "s", "questions": {}, "gate": 0.5, "return_details": True})


if __name__ == "__main__":
    unittest.main()
