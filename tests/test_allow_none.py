"""allow_none: the opt-in "None of the above" option of choice questions. Off (the default) must change nothing; on adds
one last option to the prompt, reads n + 1 options, and answers with renormalised probabilities, none_probability and
a null choice when the extra option is the most likely."""
from __future__ import annotations

import asyncio
import math
import re
import unittest
from dataclasses import replace

from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls, RequestError, parse_controls
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import NONE_OPTION, from_systemone, messages, option_lines
from startlux_decide.readout import choice_answer, split_none

from .test_pipeline import AUTO, INSTANT, QUESTIONS, closing

DEFAULTS = Controls()
TEAM = QUESTIONS["team"]                                        # billing, shipping
ONLY = {"q": {"type": "choice", "instructions": "q", "criteria": {"only": None}}}     # a choice with one option


def run(coro):
    return asyncio.run(coro)


def with_none(name="team", **extra):
    return {name: dict(TEAM, **extra)}


class Scripted(FakeBackend):
    """A fake engine whose read-outs follow `probs(prompt, k)` (probabilities of the k letters, or None for the
    deterministic hash read-out)."""

    def __init__(self, probs, **kw):
        super().__init__(**kw)
        self.probs = probs

    async def readout(self, prompt, letter_ids, media=None):
        p = self.probs(prompt, len(letter_ids))
        if p is None:
            return await super().readout(prompt, letter_ids, media)
        assert len(p) == len(letter_ids) and abs(sum(p) - 1) < 1e-9, p
        self.calls.append(("readout", prompt, tuple(letter_ids)))
        return [math.log(v) for v in p]

    def readouts(self):
        return [c for c in self.calls if c[0] == "readout"]


def fixed(*p):
    """Every read-out of a prompt that holds the extra option gives p; the others the hash read-out."""
    return lambda prompt, k: list(p) if NONE_OPTION in prompt and k == len(p) else None


def decide(backend, questions, folder=AUTO, state="charged twice", **ctl):
    ctl.setdefault("think_mode", "never")
    return run(Decider(folder, backend).decide(state, questions, Controls(**ctl)))


class ControlsTest(unittest.TestCase):
    def test_parse(self):
        parse = lambda body: parse_controls(body, DEFAULTS, 4096, True)      # noqa: E731
        self.assertIs(DEFAULTS.allow_none, False)
        self.assertEqual(parse({}), DEFAULTS)
        self.assertIs(parse({"allow_none": True}).allow_none, True)
        self.assertIs(parse({"allow_none": False}).allow_none, False)
        self.assertEqual(parse({"allow_none": None}), DEFAULTS)                # null = not given
        self.assertEqual(parse_controls({"allow_none": False}, replace(DEFAULTS, allow_none=True), 4096, True),
                         DEFAULTS)
        for bad in ("yes", "true", 1, 0, [True], {}):
            with self.assertRaises(RequestError, msg=repr(bad)) as cm:
                parse({"allow_none": bad})
            self.assertEqual(cm.exception.loc, ["body", "allow_none"])


class PromptTest(unittest.TestCase):
    def test_extra_option_is_last_and_rendered_like_any_option(self):
        row = from_systemone("charged twice", TEAM, allow_none=True)
        self.assertEqual(option_lines(row), ["A) billing: Payments", "B) shipping: Delivery", "C) None of the above"])
        self.assertEqual(row["options"][-1], {"id": NONE_OPTION, "criterion": None})
        self.assertEqual(messages(row)[1]["content"],
                         "Evidence:\ncharged twice\n\nQuestion: Which team?\nOptions:\nA) billing: Payments\n"
                         "B) shipping: Delivery\nC) None of the above")

    def test_off_is_the_old_prompt(self):
        for row in (from_systemone("charged twice", TEAM), from_systemone("charged twice", TEAM, allow_none=False)):
            self.assertNotIn("none_option", row)
            self.assertEqual(messages(row)[1]["content"],
                             "Evidence:\ncharged twice\n\nQuestion: Which team?\nOptions:\nA) billing: Payments\n"
                             "B) shipping: Delivery")

    def test_bare_ids_are_still_hidden_for_the_listed_options(self):
        spec = {"instructions": "Capital?", "criteria": {"A": "Paris", "B": "Berlin"}}
        self.assertEqual(option_lines(from_systemone("s", spec)), ["A) Paris", "B) Berlin"])
        self.assertEqual(option_lines(from_systemone("s", spec, allow_none=True)),
                         ["A) Paris", "B) Berlin", "C) None of the above"])

    def test_ids_without_descriptions(self):
        row = from_systemone("s", {"instructions": "q", "criteria": ["x", "y"]}, allow_none=True)
        self.assertEqual(option_lines(row), ["A) x", "B) y", "C) None of the above"])

    def test_only_choice_questions_get_it(self):
        for spec in ({"type": "noul", "instructions": "q"}, QUESTIONS["severity"]):
            self.assertEqual(from_systemone("s", spec, allow_none=True), from_systemone("s", spec))

    def test_an_option_of_that_name_is_refused(self):
        for name in ("None of the above", "none of the above", "None_of_the_above", " None-of-the-above "):
            with self.assertRaises(ValueError, msg=name):
                from_systemone("s", {"instructions": "q", "criteria": {"a": None, name: None}}, allow_none=True)
        from_systemone("s", {"instructions": "q", "criteria": {"a": None, "None of the above": None}})   # off: fine


class ReadoutTest(unittest.TestCase):
    def test_split_none(self):
        listed, none, wins = split_none([0.2, 0.1, 0.7])
        self.assertAlmostEqual(listed[0], 2 / 3)
        self.assertAlmostEqual(listed[1], 1 / 3)
        self.assertEqual((none, wins), (0.7, True))
        self.assertFalse(split_none([0.4, 0.2, 0.4])[2])                        # a tie goes to the listed option
        self.assertEqual(split_none([0.0, 0.0, 1.0]), ([0.5, 0.5], 1.0, True))   # nothing left to renormalise
        self.assertEqual(choice_answer({"a": 1.0}), {"type": "choice", "choice": "a", "confidence": 1.0,
                                                     "probabilities": {"a": 1.0}})


class OffTest(unittest.TestCase):
    def test_off_changes_nothing(self):
        runs = []
        for ctl, questions in ((Controls(), QUESTIONS), (Controls(allow_none=False), QUESTIONS),
                               (Controls(), {**QUESTIONS, "team": dict(TEAM, allow_none=False)}),
                               (Controls(), {**QUESTIONS, "team": dict(TEAM, allow_none=None)})):
            backend = FakeBackend(script=closing)
            out = run(Decider(AUTO, backend).decide("charged twice", questions, replace(ctl, think_mode="always")))
            runs.append((out.answers, out.usage, backend.calls))
        for other in runs[1:]:
            self.assertEqual(other, runs[0])
        answers = runs[0][0]
        self.assertEqual(list(answers["team"]), ["type", "choice", "confidence", "probabilities"])
        self.assertNotIn(NONE_OPTION, "".join(c[1] for c in runs[0][2]))

    def test_off_keeps_the_single_option_shortcut(self):
        backend = FakeBackend()
        out = decide(backend, ONLY)
        self.assertEqual(out.answers["q"], {"type": "choice", "choice": "only", "confidence": 1.0,
                                            "probabilities": {"only": 1.0}})
        self.assertEqual(backend.calls, [])


class OnTest(unittest.TestCase):
    def test_prompt_has_the_extra_last_option(self):
        backend = Scripted(fixed(0.5, 0.3, 0.2))
        decide(backend, with_none(), allow_none=True)
        (call,) = backend.readouts()
        self.assertIn("Options:\nA) billing: Payments\nB) shipping: Delivery\nC) None of the above<|im_end|>", call[1])
        self.assertEqual(len(call[2]), 3)                                       # three letters read

    def test_best_listed_option(self):
        out = decide(Scripted(fixed(0.5, 0.3, 0.2)), with_none(), allow_none=True)
        a = out.answers["team"]
        self.assertEqual(list(a), ["type", "choice", "confidence", "probabilities", "none_probability"])
        self.assertEqual(a["choice"], "billing")
        self.assertAlmostEqual(a["none_probability"], 0.2)
        self.assertEqual(list(a["probabilities"]), ["billing", "shipping"])        # the listed options only
        self.assertAlmostEqual(a["probabilities"]["billing"], 0.5 / 0.8)
        self.assertAlmostEqual(a["probabilities"]["shipping"], 0.3 / 0.8)
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)
        self.assertAlmostEqual(a["confidence"], (2 * (0.5 / 0.8) - 1) / 1)       # from the renormalised ones
        self.assertEqual(out.usage["output_tokens"], 0)

    def test_none_is_the_most_likely(self):
        a = decide(Scripted(fixed(0.2, 0.1, 0.7)), with_none(), allow_none=True).answers["team"]
        self.assertIsNone(a["choice"])
        self.assertAlmostEqual(a["none_probability"], 0.7)
        self.assertAlmostEqual(a["probabilities"]["billing"], 2 / 3)
        self.assertAlmostEqual(a["probabilities"]["shipping"], 1 / 3)
        self.assertAlmostEqual(a["confidence"], 1 / 3)

    def test_a_tie_goes_to_the_listed_option(self):
        a = decide(Scripted(fixed(0.4, 0.2, 0.4)), with_none(), allow_none=True).answers["team"]
        self.assertEqual(a["choice"], "billing")

    def test_single_option_is_read_by_the_model(self):
        backend = Scripted(fixed(0.3, 0.7))
        out = decide(backend, ONLY, allow_none=True)
        a = out.answers["q"]
        self.assertIsNone(a["choice"])
        self.assertEqual(a["probabilities"], {"only": 1.0})
        self.assertAlmostEqual(a["none_probability"], 0.7)
        (call,) = backend.readouts()
        self.assertIn("Options:\nA) only\nB) None of the above<|im_end|>", call[1])
        a = decide(Scripted(fixed(0.7, 0.3)), ONLY, allow_none=True).answers["q"]
        self.assertEqual((a["choice"], a["probabilities"]), ("only", {"only": 1.0}))

    def test_instant_models_too(self):
        a = decide(Scripted(fixed(0.5, 0.3, 0.2)), with_none(), folder=INSTANT, allow_none=True).answers["team"]
        self.assertEqual((a["choice"], round(a["none_probability"], 6)), ("billing", 0.2))

    def test_request_and_question_settings(self):
        questions = {"a": dict(TEAM), "b": dict(TEAM, allow_none=False), "c": dict(TEAM, allow_none=True),
                     "d": dict(TEAM, allow_none=None)}
        prompts = lambda backend: backend.readouts()                                        # noqa: E731
        for request, expected in ((True, {"a": True, "b": False, "c": True, "d": True}),
                                  (False, {"a": False, "b": False, "c": True, "d": False})):
            backend = Scripted(lambda prompt, k: None)
            out = decide(backend, questions, allow_none=request)
            self.assertEqual({k: "none_probability" in a for k, a in out.answers.items()}, expected)
            self.assertEqual({k: len(a["probabilities"]) for k, a in out.answers.items()}, {k: 2 for k in questions})
            self.assertEqual(sorted(len(c[2]) for c in prompts(backend)),
                             sorted(3 if v else 2 for v in expected.values()))

    def test_question_setting_must_be_a_bool(self):
        for bad in ("yes", 1, 0, []):
            with self.assertRaises(RequestError, msg=repr(bad)) as cm:
                decide(FakeBackend(), {"ok": QUESTIONS["urgent"], "team": dict(TEAM, allow_none=bad)})
            self.assertEqual(cm.exception.loc, ["body", "questions", "team", "allow_none"])

    def test_other_question_types_ignore_it(self):
        questions = {"urgent": QUESTIONS["urgent"], "severity": QUESTIONS["severity"]}
        plain, on = FakeBackend(), FakeBackend()
        a = decide(plain, questions).answers
        b = decide(on, {k: dict(v, allow_none=True) for k, v in questions.items()}, allow_none=True).answers
        c = decide(FakeBackend(), {k: dict(v, allow_none="junk") for k, v in questions.items()}).answers   # not checked
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(plain.calls, on.calls)
        self.assertNotIn(NONE_OPTION, "".join(c[1] for c in on.calls))

    def test_an_option_called_none_of_the_above_is_refused(self):
        q = {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None, "None of the above": None}}}
        with self.assertRaises(RequestError) as cm:
            decide(FakeBackend(), q, allow_none=True)
        self.assertEqual(cm.exception.loc, ["body", "questions", "q"])
        self.assertIn("allow_none", str(cm.exception))
        decide(FakeBackend(), q)                                                    # off: an ordinary option

    def test_details_show_the_read_outs_over_all_options(self):
        out = decide(Scripted(fixed(0.5, 0.3, 0.2)), with_none(), allow_none=True, details=True)
        det = out.answers["team"]["details"]
        self.assertEqual(det["first_pass"]["probabilities"].keys(), {"billing", "shipping", NONE_OPTION})
        self.assertAlmostEqual(det["first_pass"]["probabilities"][NONE_OPTION], 0.2)
        self.assertAlmostEqual(det["first_pass"]["probabilities"]["billing"], 0.5)
        self.assertAlmostEqual(det["first_pass"]["top"], 0.5)
        self.assertIsNone(det["after_thought"])

    def test_the_thought_and_the_read_out_after_it_see_the_option(self):
        def probs(prompt, k):
            if k != 3:
                return None
            return [0.1, 0.1, 0.8] if "Billing handles" in prompt else [0.2, 0.3, 0.5]
        backend = Scripted(probs, script=closing)
        out = decide(backend, with_none(), allow_none=True, think_mode="always", details=True)
        a = out.answers["team"]
        gen = [c for c in backend.calls if c[0] == "generate"]
        self.assertTrue(gen and all("C) None of the above" in c[1] for c in gen))
        det = a["details"]
        self.assertTrue(det["thought"]["used"])
        self.assertEqual(det["after_thought"]["probabilities"].keys(), {"billing", "shipping", NONE_OPTION})
        # (1 - mix) * first + mix * after, mix = 0.75, then split
        mixed = [0.25 * x + 0.75 * y for x, y in zip([0.2, 0.3, 0.5], [0.1, 0.1, 0.8])]
        self.assertAlmostEqual(a["none_probability"], mixed[2])
        self.assertAlmostEqual(a["probabilities"]["billing"], mixed[0] / (mixed[0] + mixed[1]))
        self.assertIsNone(a["choice"])                                              # 0.725 beats 0.125 and 0.15
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)


def final_round(none, weight=lambda i: 1.0):
    """Read-outs of the prompt that holds the extra option: `none` on the last letter, the rest shared out by weight."""
    def probs(prompt, k):
        if NONE_OPTION not in prompt:
            return None
        w = [weight(i) for i in range(k - 1)]
        return [(1 - none) * v / sum(w) for v in w] + [none]
    return probs


class KnockoutTest(unittest.TestCase):
    def wide(self, n):
        return {"q": {"type": "choice", "instructions": "Which?", "criteria": {f"opt{i}": f"option number {i}"
                                                                                for i in range(n)}}}

    def test_extra_option_only_in_the_final_round(self):
        backend = Scripted(final_round(0.6))
        out = decide(backend, self.wide(60), allow_none=True, details=True)
        self.assertEqual([len(c[2]) for c in backend.readouts()], [20, 20, 20, 10])        # 9 finalists + none
        self.assertEqual([NONE_OPTION in c[1] for c in backend.readouts()], [False, False, False, True])
        a = out.answers["q"]
        self.assertIsNone(a["choice"])
        self.assertAlmostEqual(a["none_probability"], 0.6)
        self.assertEqual(len(a["probabilities"]), 60)
        self.assertNotIn(NONE_OPTION, a["probabilities"])
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)
        shown = a["details"]["first_pass"]["probabilities"]
        self.assertEqual(len(shown), 61)
        self.assertAlmostEqual(shown[NONE_OPTION], 0.6)
        self.assertAlmostEqual(sum(shown.values()), 1.0)

    def test_best_finalist_when_none_is_unlikely(self):
        backend = Scripted(final_round(0.1, weight=lambda i: 5.0 if i == 4 else 1.0))
        a = decide(backend, self.wide(60), allow_none=True).answers["q"]
        final_prompt = backend.readouts()[-1][1]
        shown = re.findall(r"^[A-Z]\) option number (\d+)$", final_prompt, re.M)       # bare ids are hidden
        self.assertEqual(len(shown), 9)
        self.assertEqual(a["choice"], f"opt{shown[4]}")
        self.assertAlmostEqual(a["none_probability"], 0.1)
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)

    def test_off_is_the_old_knockout(self):
        backend = Scripted(lambda prompt, k: None)
        out = decide(backend, self.wide(60))
        self.assertEqual([len(c[2]) for c in backend.readouts()], [20, 20, 20, 9])
        self.assertNotIn("none_probability", out.answers["q"])
        self.assertNotIn(NONE_OPTION, "".join(c[1] for c in backend.readouts()))

    def test_the_letters_the_extra_option_needs(self):
        for n, on, off in ((24, [25], [24]), (25, [26], [25]), (26, [13, 13, 7], [26]), (27, [14, 13, 7], [14, 13, 6])):
            for allow, lens in ((True, on), (False, off)):
                backend = Scripted(final_round(0.3) if allow else (lambda prompt, k: None))
                decide(backend, self.wide(n), allow_none=allow)
                self.assertEqual([len(c[2]) for c in backend.readouts()], lens, (n, allow))

    def test_final_round_holds_at_most_25_listed_options(self):
        for allow in (True, False):
            backend = Scripted(final_round(0.3) if allow else (lambda prompt, k: None))
            decide(backend, self.wide(300), allow_none=allow)
            last = backend.readouts()[-1]
            self.assertEqual(len(last[2]), 26)                      # 25 listed options and the extra one | 26 listed
            self.assertEqual(NONE_OPTION in last[1], allow)


if __name__ == "__main__":
    unittest.main()
