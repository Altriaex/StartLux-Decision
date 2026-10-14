"""none_check: the internal "None of the above" option, on by default in `serve`. Every choice answer then carries
none_probability, but "choice" stays a listed option; only allow_none lets it be null."""
from __future__ import annotations

import unittest

from startlux_decide import cli
from startlux_decide.controls import Controls, RequestError, parse_controls
from startlux_decide.pipeline import default_controls

from .test_allow_none import ONLY, Scripted, decide, fixed, with_none
from .test_pipeline import AUTO, INSTANT


class NoneCheckTest(unittest.TestCase):
    def test_parse(self):
        self.assertFalse(Controls().none_check)
        self.assertTrue(parse_controls({"none_check": True}, Controls(), 4096, True).none_check)
        self.assertFalse(parse_controls({"none_check": False}, Controls(none_check=True), 4096, True).none_check)
        with self.assertRaises(RequestError):
            parse_controls({"none_check": "yes"}, Controls(), 4096, True)

    def test_server_default(self):
        self.assertTrue(default_controls(AUTO, none_check=True).none_check)
        self.assertFalse(default_controls(AUTO).none_check)

    def test_server_allow_none_default(self):
        d = default_controls(AUTO, none_check=True, allow_none=True)
        self.assertTrue(d.allow_none)
        self.assertFalse(default_controls(AUTO, none_check=True).allow_none)
        self.assertFalse(parse_controls({"allow_none": False}, d, 4096, True).allow_none)
        questions = {"a": with_none()["team"], "b": dict(with_none()["team"], allow_none=False)}
        out = decide(Scripted(fixed(0.2, 0.1, 0.7)), questions, none_check=True, allow_none=True)
        self.assertEqual({k: a["choice"] for k, a in out.answers.items()}, {"a": None, "b": "billing"})
        self.assertTrue(cli.build_parser().parse_args(["serve", "--model", "m", "--allow-none"]).allow_none)
        self.assertFalse(cli.build_parser().parse_args(["serve", "--model", "m"]).allow_none)

    def test_listed_option_even_when_none_is_most_likely(self):
        a = decide(Scripted(fixed(0.2, 0.1, 0.7)), with_none(), none_check=True).answers["team"]
        self.assertEqual(a["choice"], "billing")                        # not null: allow_none is off
        self.assertAlmostEqual(a["none_probability"], 0.7)
        self.assertAlmostEqual(a["probabilities"]["billing"], 2 / 3)
        self.assertAlmostEqual(a["confidence"], 1 / 3)

    def test_allow_none_still_gives_null(self):
        a = decide(Scripted(fixed(0.2, 0.1, 0.7)), with_none(), none_check=True, allow_none=True).answers["team"]
        self.assertIsNone(a["choice"])

    def test_question_allow_none_on_top_of_the_check(self):
        questions = {"a": with_none()["team"], "b": dict(with_none()["team"], allow_none=True),
                     "c": dict(with_none()["team"], allow_none=False)}
        out = decide(Scripted(fixed(0.2, 0.1, 0.7)), questions, none_check=True)
        self.assertEqual({k: a["choice"] for k, a in out.answers.items()}, {"a": "billing", "b": None, "c": "billing"})
        self.assertTrue(all("none_probability" in a for a in out.answers.values()))

    def test_same_answer_as_allow_none_when_a_listed_option_wins(self):
        p = (0.5, 0.3, 0.2)
        a = decide(Scripted(fixed(*p)), with_none(), none_check=True).answers["team"]
        b = decide(Scripted(fixed(*p)), with_none(), allow_none=True).answers["team"]
        self.assertEqual(a, b)

    def test_off_changes_nothing(self):
        a = decide(Scripted(lambda prompt, k: None), with_none()).answers["team"]
        self.assertNotIn("none_probability", a)

    def test_single_option_and_instant(self):
        a = decide(Scripted(fixed(0.3, 0.7)), ONLY, none_check=True).answers["q"]
        self.assertEqual((a["choice"], a["probabilities"]), ("only", {"only": 1.0}))
        self.assertAlmostEqual(a["none_probability"], 0.7)
        a = decide(Scripted(fixed(0.5, 0.3, 0.2)), with_none(), folder=INSTANT, none_check=True).answers["team"]
        self.assertEqual(a["choice"], "billing")


if __name__ == "__main__":
    unittest.main()
