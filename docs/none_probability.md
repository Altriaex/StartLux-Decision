# none_probability

Every `choice` answer of `startlux-decide serve` carries `none_probability`: how strongly the model believes that
none of the listed options is right. This page explains where the number comes from, how to read it, why we suggest
0.2 as the threshold, and where it is not reliable.

## Where it comes from

The model answers a choice question by reading the probability of each option's letter at one position of the
prompt; it does not generate text. With `none_check` (on by default in `serve`) the engine adds one more option at the
end of the list, "None of the above", with its own letter, and the model scores it like any other option.
`none_probability` is the share of the read-out that goes to that extra letter. The listed options' probabilities are
then scaled up to sum to 1, so `probabilities` keeps its usual meaning, and `choice` is the most likely listed option.
With `"allow_none": true`, `choice` is null when "None of the above" is the most likely option.

For an AutoThink answer that went through a thought, the final probabilities mix the read-out before the thought and
the one after it (0.25 and 0.75 by default), and `none_probability` is the mixed value. The extra option also takes
part in the confidence gate: a question whose read-out gives a lot to "None of the above" has no listed option above
the gate and therefore gets a thought.

An example. Asked what 1 + 1 is, with only 5 and 10 as options, the model gives about 0.02 to 5, 0.01 to 10 and 0.97 to
"None of the above". The answer is `none_probability` 0.97, `probabilities` {5: 0.67, 10: 0.33} and `choice` 5 (null
with `allow_none`).

The models were not trained on questions with this extra option. `none_probability` is how they read the words "None
of the above" on their own, which is why it works well as a signal but is not a calibrated probability.

## How we measured it

We took every choice question with 3 to 24 options and a known right answer from DecisionPivot (6,007 questions) and
the public part of Decision Index 0.3 (45,499 questions), leaving out questions whose options already offered "none",
"other" or similar. Each question was asked twice with the extra option: as published, and with its right answer
removed, so that no listed option was right. The model was StartLux-Decision-v1.1-27B-AutoThink on vLLM, with thinking
on (`think_mode` `auto`) unless a row says otherwise. All numbers below come from that run of 51,506 questions; the
smaller models were not measured.

Adding the extra option does not cost accuracy: on the published questions the most likely listed option was right
84.66% of the time with the extra option and 84.66% without it (78.53% and 78.50% in one pass).

## How to read it

Read `none_probability` as an alarm score, not as an exact probability. The model is reluctant to pick "None of the
above", so its numbers run low. In a set made half of published questions and half of questions without a right
answer, this is the share of questions that really had no right answer, by `none_probability`:

| `none_probability` | 0.05 to 0.1 | 0.1 to 0.2 | 0.2 to 0.3 | 0.3 to 0.5 | 0.5 and above |
|---|---|---|---|---|---|
| no right answer | 56% | 77% | 87% | 93% | 96% to 99% |

As a ranking it is strong: given one question without a right answer and one with, the first has the higher
`none_probability` 92% of the time (AUROC 0.920; 0.892 in one pass).

## Why 0.2

| Threshold | Questions without a right answer caught | Published questions flagged by mistake |
|---|---|---|
| 0.1 | 73.7% | 4.5% |
| **0.2** | **66.3%** | **2.2%** |
| 0.3 | 61.5% | 1.6% |
| 0.5 | 53.5% | 1.0% |

We suggest treating `none_probability` ≥ 0.2 as "no listed option fits" for four reasons.

1. **False alarms stay near 2%.** At 0.2 about one well-posed question in 45 is flagged. Going down to 0.1 catches
   7 points more but doubles the false alarms to 4.5%. Going up to 0.5 halves them to 1.0% but misses 13 points more.
2. **Above 0.2 a flag is usually right.** Questions between 0.2 and 0.3 had no right answer 87% of the time in the
   half-and-half set; between 0.1 and 0.2 it was 77%, and between 0.05 and 0.1 close to a coin toss (56%).
3. **It holds across settings.** The lowest threshold that keeps false alarms under 3% is 0.15 with thinking and 0.19 in
   one pass; 0.2 keeps both under 3% (2.2% and 2.6%), and false alarms are 2.5% on DecisionPivot and 2.2% on Decision
   Index.
4. **Lower cuts flag too many good questions.** The best single cut by Youden's J is lower (0.06), but around it about
   one published question in ten is flagged (10.4% at 0.05), which is too many for most applications.

What a flag means also depends on how often your traffic has no right answer. Share of flagged questions that really
had no right answer:

| Questions without a right answer in your traffic | 1% | 5% | 20% |
|---|---|---|---|
| threshold 0.1 | 14% | 46% | 80% |
| threshold 0.2 | 23% | 61% | 88% |
| threshold 0.5 | 35% | 74% | 93% |

Use 0.1 when a missed problem is expensive and a person reviews every flag, 0.5 when a flag interrupts a user, and 0.2
otherwise. If you can, set the threshold on a few hundred of your own questions.

## Where it is not reliable

It works best when a question has one answer that can be checked against the state: rule and policy decisions,
extraction, fact checks, entailment, arithmetic. There 77% of the questions without a right answer are caught at 0.2,
with 1.5% false alarms.

It is weak in two situations, and a threshold cannot fix either:

- **When "none of them" is not well defined.** For preferences between acceptable options (which plan, which
  consensus statement people would choose) and for the choice of a move in a game, removing the best option still
  leaves a best option, and the model does not call the rest wrong. Detection was 0% to 13% on these tasks. Comparisons
  with a clear right answer are not affected: picking the better of two responses to a prompt was caught 83% of the
  time.
- **When the model does not know the answer.** On very hard questions (graduate-level science, expert exams) the model
  cannot tell that the right answer is missing: 16% to 47% were caught.

False alarms are higher than average where options are unusual or long: matching captions to cartoons (25%), choosing
an element on a web page (14%) and long multi-step reasoning stories (12%).

In these settings add your own "none of these" option to the list, or check the answer another way.

## Thinking

Questions without a right answer make AutoThink think more often: 78% of them got a thought against 51% of published
questions, and their thoughts were longer (544 against 290 thought tokens per question on average). Thinking also helps
the detection: 66.3% caught at 0.2 against 57.4% in one pass.

## Using it

```python
answer = response["answers"]["route"]
if answer["none_probability"] >= 0.2:
    escalate_to_a_person(answer)          # or ask again with more options
else:
    act_on(answer["choice"], answer["confidence"])
```

`none_check` adds one letter to every choice read-out and nothing else. Send `"none_check": false` in a request, or start
the server with `--no-none-check`, to leave the extra option out; answers then carry no `none_probability`. Send
`"allow_none": true` to get `choice: null` when "None of the above" wins. A server started with `--allow-none` does
that by default, and a request or a question can still send `"allow_none": false`.
