"""Probabilities, re-tempering, mixture, confidences, answers and the knock-out of long choice lists."""
from __future__ import annotations

import math
import unittest

from startlux_decide.prompt import from_systemone
from startlux_decide.readout import (answer, choice_confidence, knockout_combine, knockout_finalists, knockout_groups,
                                     mixture, option_probabilities, probabilities, retemper, score_confidence)


class ProbabilityTest(unittest.TestCase):
    def test_softmax_with_temperature(self):
        p = probabilities([0.0, math.log(3.0)], 1.0)
        self.assertAlmostEqual(p[1], 0.75)
        self.assertEqual(probabilities([1.0, 2.0, 3.0], 1.3), probabilities([11.0, 12.0, 13.0], 1.3))
        self.assertGreater(max(probabilities([0, 1, 2], 0.5)), max(probabilities([0, 1, 2], 2.0)))
        self.assertEqual(probabilities([-1e9, 0.0], 1.0), [0.0, 1.0])

    def test_retemper(self):
        p = probabilities([0.0, 1.0, 2.5], 1.25)
        for a, b in zip(retemper(p, 1.25, 1.0), probabilities([0.0, 1.0, 2.5], 1.0)):
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(retemper(p, 1.0, 1.0), p):
            self.assertAlmostEqual(a, b, places=15)
        self.assertEqual(sum(retemper([1.0, 0.0], 2.0, 1.0)), 1.0)        # zeros are floored, not an error

    def test_mixture_ends(self):
        p0, p1 = [0.2, 0.8], [0.9, 0.1]
        self.assertEqual(mixture(p0, p1, 0.0), p0)
        self.assertEqual(mixture(p0, p1, 1.0), p1)
        self.assertEqual(mixture(p0, p1, 0.75), [0.25 * 0.2 + 0.75 * 0.9, 0.25 * 0.8 + 0.75 * 0.1])


class ConfidenceTest(unittest.TestCase):
    def test_choice(self):
        self.assertEqual(choice_confidence([0.25] * 4), 0.0)
        self.assertEqual(choice_confidence([1.0, 0.0, 0.0]), 1.0)
        self.assertAlmostEqual(choice_confidence([0.6, 0.4]), 0.2)
        self.assertEqual(choice_confidence([1.0]), 1.0)

    def test_score(self):
        self.assertAlmostEqual(score_confidence([0.8, 0.2]), abs(2 * 0.8 - 1))    # two levels: |2p - 1|
        self.assertEqual(score_confidence([0.0, 1.0, 0.0]), 1.0)
        self.assertEqual(score_confidence([1 / 3] * 3), 0.0)
        near, far = score_confidence([0.0, 0.7, 0.3, 0.0]), score_confidence([0.3, 0.7, 0.0, 0.0])
        self.assertEqual(near, far)                          # the same distance of one level either side
        self.assertGreater(score_confidence([0.7, 0.3, 0.0, 0.0]), score_confidence([0.7, 0.0, 0.0, 0.3]))


class AnswerTest(unittest.TestCase):
    def test_formats(self):
        noul = from_systemone("s", {"type": "noul"})
        self.assertEqual(answer(noul, [0.3, 0.7], {"type": "noul"}), {"type": "noul", "noul": 0.3})
        spec = {"type": "score", "instructions": "q", "criteria": {"1": "mid", "0": "low", "2": "high"}}
        a = answer(from_systemone("s", spec), [0.2, 0.3, 0.5], spec)
        self.assertEqual(a["legend"], {"0": "low", "1": "mid", "2": "high"})
        self.assertAlmostEqual(a["score"], 1.3)
        self.assertEqual(a["probabilities"], {"0": 0.2, "1": 0.3, "2": 0.5})
        spec = {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": None}}
        a = answer(from_systemone("s", spec), [0.5, 0.5], spec)
        self.assertEqual((a["choice"], a["confidence"]), ("x", 0.0))       # ties go to the option listed first
        self.assertEqual(option_probabilities(noul, [0.3, 0.7]), {"true": 0.3, "false": 0.7})


class KnockoutTest(unittest.TestCase):
    def test_groups(self):
        self.assertEqual([len(g) for g in knockout_groups([str(i) for i in range(27)], 25)], [14, 13])
        self.assertEqual([len(g) for g in knockout_groups([str(i) for i in range(255)], 25)], [24] * 2 + [23] * 9)
        keys = [str(i) for i in range(100)]
        self.assertEqual(sum(knockout_groups(keys, 25), []), keys)

    def test_finalists_and_overflow(self):
        groups = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
        first, fin = knockout_finalists(groups, [[0.1, 0.4, 0.3, 0.2], [0.7, 0.1, 0.1, 0.1]], 3, 26)
        self.assertEqual(fin, ["b", "c", "d", "e", "f", "g"])                  # group order, best first in a group
        _, fin = knockout_finalists(groups, [[0.1, 0.4, 0.3, 0.2], [0.7, 0.1, 0.1, 0.1]], 3, 4)
        self.assertEqual(fin, ["e", "b", "c", "d"])                            # over the limit: best overall

    def test_combine(self):
        first = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
        dist = knockout_combine({"a": 0.6, "b": 0.4}, first, 0.001)
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertEqual(list(dist), ["a", "b", "c", "d"])
        self.assertAlmostEqual(dist["c"] / dist["d"], 3.0)
        self.assertAlmostEqual(dist["c"] + dist["d"], 0.001)


if __name__ == "__main__":
    unittest.main()
