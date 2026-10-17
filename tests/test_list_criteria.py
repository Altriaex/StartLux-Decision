"""Every form of a choice list is knocked out alike: {option: description}, a list of options, or either under
"options". A list of options is read as from_systemone reads it ({str(option): None}), so its answers, prompts and
engine calls are those of that dict; before, a list of more than 26 options (or of 26 with the extra "None of the
above") was refused instead of knocked out."""
from __future__ import annotations

import asyncio
import unittest

from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls
from startlux_decide.pipeline import Decider, choice_criteria

from .test_pipeline import AUTO, INSTANT


def run(coro):
    return asyncio.run(coro)


def decide(folder, question, **ctl):
    backend = FakeBackend()
    out = run(Decider(folder, backend).decide("a state", {"q": question}, Controls(think_mode="never", **ctl)))
    return out.answers["q"], out.usage, backend.calls


class ListCriteriaTest(unittest.TestCase):
    def same_as_dict(self, ids, folder=AUTO, **ctl):
        """The list form, the "options" forms and the dict of None descriptions give the same answer and calls."""
        want = decide(folder, {"type": "choice", "instructions": "Which?", "criteria": {str(i): None for i in ids}},
                      **ctl)
        for q in ({"criteria": ids}, {"options": ids}, {"options": {str(i): None for i in ids}}):
            with self.subTest(form=list(q), n=len(ids), **ctl):
                self.assertEqual(decide(folder, dict(q, type="choice", instructions="Which?"), **ctl), want)
        return want

    def test_long_lists_are_knocked_out(self):
        for n in (27, 40, 300):
            ans, usage, calls = self.same_as_dict([f"opt{i}" for i in range(n)])
            self.assertEqual(len(ans["probabilities"]), n)
            self.assertGreater(len(calls), 1)                          # group rounds and a final round

    def test_26_with_the_extra_option(self):
        ids = [f"opt{i}" for i in range(26)]
        for ctl in ({"none_check": True}, {"allow_none": True}):
            ans, _, calls = self.same_as_dict(ids, INSTANT, **ctl)
            self.assertIn("none_probability", ans)
            self.assertEqual(len(ans["probabilities"]), 26)
            self.assertEqual(len(calls), 3)                            # two groups of 13, then the final
        ans, _, calls = self.same_as_dict(ids)                         # without it: one pass, as before
        self.assertEqual(len(calls), 1)

    def test_short_lists_unchanged(self):
        for ids in (["a", "b"], ["billing", "shipping", "refund"], [1, 2, 3], ["1", 2]):
            _, _, calls = self.same_as_dict(ids)
            self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1].count("\nA) 1\nB) 2<|im_end|>"))

    def test_one_option_and_duplicates(self):
        ans, usage, calls = self.same_as_dict(["only"])                # answered without the model, as a dict is
        self.assertEqual((ans["choice"], ans["probabilities"], calls), ("only", {"only": 1.0}, []))
        ans, _, _ = self.same_as_dict([1, "1", 2])                     # one id after str(), as from_systemone reads it
        self.assertEqual(set(ans["probabilities"]), {"1", "2"})

    def test_choice_criteria(self):
        self.assertEqual(choice_criteria({"criteria": [1, "b"]}), {"1": None, "b": None})
        self.assertEqual(choice_criteria({"options": ("x",)}), {"x": None})
        d = {"a": "A"}
        self.assertIs(choice_criteria({"criteria": d, "options": ["z"]}), d)
        self.assertIsNone(choice_criteria({"type": "choice"}))


if __name__ == "__main__":
    unittest.main()
