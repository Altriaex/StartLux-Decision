"""The loop detector against the release's (version 3) on generated thoughts, and the rules one by one."""
from __future__ import annotations

import random
import unittest

from startlux_decide.loops import few_lines, loop_cut, looping, low_diversity

from . import reference_impl as ref
from .fake_vllm import WORDS


def generated_text(r: random.Random) -> str:
    """Prose, optionally followed by one kind of loop, cut at a random length."""
    prose = "".join(r.choice(WORDS) + r.choice([" ", " ", " ", "\n", ", "]) for _ in range(r.randint(0, 400)))
    kind = r.choice(["none", "short", "short_digits", "long", "digits", "lines", "words", "lines_numbered", "near"])
    if kind == "short":
        tail = r.choice(["B. ", "A\n", "The answer is C.\n", "x"]) * r.randint(1, 60)
    elif kind == "short_digits":
        tail = "".join(f"Step {i}: check.\n" for i in range(r.randint(1, 40)))
    elif kind == "long":
        unit = " ".join(r.choice(WORDS) for _ in range(r.randint(15, 70))) + ".\n"
        tail = unit * r.randint(1, 6)
    elif kind == "digits":
        tail = "".join(r.choice("0123456789") for _ in range(r.randint(1, 400)))
    elif kind == "lines":
        lines = [f"B. 0.{r.randint(10, 99)}\n", "I'll output B.\n", "B is right.\n", "So B.\n"][:r.randint(1, 4)]
        tail = "".join(r.choice(lines) for _ in range(r.randint(1, 40)))
    elif kind == "lines_numbered":
        tail = "".join(f"2026-03-{d:02d}: row {d}\n" for d in range(1, r.randint(2, 31)))
    elif kind == "words":
        vocab = r.sample(["B", "A", "so", "is", "the", "answer", "yes", "no"], r.randint(2, 7))
        tail = " ".join(r.choice(vocab) for _ in range(r.randint(1, 120)))
    elif kind == "near":            # a loop broken just before the end
        tail = "B. " * r.randint(10, 40) + r.choice(WORDS)
    else:
        tail = ""
    text = prose + tail
    return text[:r.randint(0, len(text))] if r.random() < 0.3 else text


class ParityTest(unittest.TestCase):
    def test_random_texts(self):
        r = random.Random(11)
        hits = 0
        for _ in range(3000):
            text = generated_text(r)
            want = ref.looping(text)
            self.assertEqual(looping(text), want, repr(text[-300:]))
            self.assertEqual(loop_cut(text), ref.loop_cut(text))
            hits += want
        self.assertGreater(hits, 800)
        self.assertLess(hits, 2200)


class RulesTest(unittest.TestCase):
    def test_short_unit(self):
        self.assertTrue(looping("Some reasoning first. " + "B. " * 30))
        self.assertFalse(looping("Some reasoning first. " + "B. " * 3))

    def test_number_that_never_ends(self):
        self.assertTrue(looping("The value is 3." + "14159265358979323846" * 8))

    def test_shuffled_lines(self):
        r = random.Random(3)
        lines = ["B. 0.92", "B. 92%", "I'll output 0.92", "Answer: B"]
        text = "Reasoning.\n" + "\n".join(r.choice(lines) for _ in range(30))
        self.assertTrue(few_lines(text))
        self.assertTrue(looping(text))

    def test_numbered_rows(self):
        # the line rule keeps digits, so rows that differ in their numbers are different lines; the periodic rules
        # replace digits, so the same rows ARE one repeated unit for them (as in the release detector)
        text = "".join(f"Row {i}: item {i * 7 % 13}, qty {i % 5}.\n" for i in range(60))
        self.assertFalse(few_lines(text))
        self.assertTrue(looping(text))
        dates = "".join(f"On 2026-03-{d:02d} the order {d * 37 % 101} was {['late', 'early', 'lost'][d % 3]}.\n"
                        for d in range(1, 29))
        self.assertFalse(few_lines(dates))

    def test_letter_soup(self):
        r = random.Random(5)
        self.assertTrue(low_diversity("Start. " + " ".join(r.choice(["B", "is", "so", "B."]) for _ in range(80))))

    def test_cut_keeps_the_reasoning_and_the_start_of_the_loop(self):
        text = "Weighing both options, the refund rule applies. " + "The answer is B. " * 40
        cut = loop_cut(text)
        self.assertTrue(text[:cut].startswith("Weighing both options, the refund rule applies. The answer is B."))
        self.assertTrue(looping(text[:cut]))
        self.assertLess(cut, len(text) // 2)
        self.assertIsNone(loop_cut("no loop here at all"))


if __name__ == "__main__":
    unittest.main()
