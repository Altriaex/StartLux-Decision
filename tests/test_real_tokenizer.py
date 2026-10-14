"""Checks against the tokenizer of a real model folder (only its tokenizer files are read). Set
STARTLUX_TEST_MODEL_DIR to a StartLux-Decision folder to run them; they are skipped otherwise.

* the letter ids match the folder's letter_token_ids, and the one-pass / thinking prompts end as trained;
* the vLLM plugin's prompt-end rule (on token ids) gives the same prompt end as the transformers reference rule
  (the last "<think>" followed by a newline, or the whole input for a one-pass prompt) on many prompts;
* the pipeline runs on the real chat template.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import unittest

from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls
from startlux_decide.modelcfg import load_model_folder, load_tokenizer
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import check_tokenizer, from_systemone, messages, plain_prompt, think_prompt
from startlux_decide.vllm_plugin import PromptEnd

from .test_golden import random_question, random_state

MODEL_DIR = os.environ.get("STARTLUX_TEST_MODEL_DIR")


def reference_prompt_end(ids, tok):
    """The transformers reference rule (the training code's): the whole input when it ends with the empty think
    block, else right after the "<think>\\n" of the last "<think>"."""
    think = tok.convert_tokens_to_ids("<think>")
    j = [i for i, t in enumerate(ids) if t == think][-1]
    if list(ids[j:]) == tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False):
        return len(ids)
    assert tok.decode(list(ids[j:j + 2])) == "<think>\n"
    return j + 2


@unittest.skipUnless(MODEL_DIR, "set STARTLUX_TEST_MODEL_DIR to a model folder")
class RealTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, "decision_config.json")) as f:
            cls.cfg = json.load(f)

    def test_letters_and_prompts(self):
        self.assertEqual(check_tokenizer(self.tok), self.cfg["letter_token_ids"])
        msgs = messages(from_systemone({"ticket": "charged twice"}, {"type": "noul", "instructions": "Refund?"}))
        self.assertTrue(plain_prompt(self.tok, msgs).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertTrue(think_prompt(self.tok, msgs).endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(plain_prompt(self.tok, msgs).startswith("<|im_start|>system\nApply the criterion"))

    def test_prompt_end_rule(self):
        pe, r = PromptEnd(self.tok), random.Random(3)
        prefix = "Let me work through this carefully, step by step.\n"
        for _ in range(300):
            spec = random_question(r)
            if isinstance(spec.get("criteria"), dict) and len(spec["criteria"]) > 26:
                continue
            msgs = messages(from_systemone(random_state(r), spec))
            thought = r.choice(["", "The refund rule applies.\nB fits.", "A stray <think>\n tag.", "x <think> y"])
            base = len(self.tok.encode(think_prompt(self.tok, msgs), add_special_tokens=False))
            plain = plain_prompt(self.tok, msgs)
            for text in (think_prompt(self.tok, msgs) + prefix + thought,
                         think_prompt(self.tok, msgs) + prefix + thought + "\n</think>\n\n"):
                ids = self.tok.encode(text, add_special_tokens=False)
                self.assertEqual(pe(ids)[0], base)
                if "<think>" not in thought:            # the reference takes the LAST <think>; a stray one in the
                    self.assertEqual(base, reference_prompt_end(ids, self.tok))   # thought would move it
            ids = self.tok.encode(plain, add_special_tokens=False)
            self.assertEqual(pe(ids), (len(ids), "one_pass"))
            self.assertEqual(reference_prompt_end(ids, self.tok), len(ids))

    def test_pipeline_on_the_real_template(self):
        folder = load_model_folder(MODEL_DIR)
        engine = FakeBackend(tokenizer=self.tok)
        d = Decider(folder, engine)
        out = asyncio.run(d.decide({"ticket": "charged twice"}, {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}},
            Controls(think_mode="never")))
        self.assertEqual(out.answers["team"]["type"], "choice")
        self.assertTrue(engine.calls[0][1].endswith("B) shipping<|im_end|>\n<|im_start|>assistant\n"
                                                   "<think>\n\n</think>\n\n"))
        self.assertGreater(out.usage["input_tokens"], 40)


if __name__ == "__main__":
    unittest.main()
