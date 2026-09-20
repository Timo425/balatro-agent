# Why the decision tree was losing, and what the numbers say now

Written after a `laya_tree` run died in Ante 1 with 216 of 300 chips, having
played four hands and used none of its four discards.

---

## 1. The chip numbers shown to the model were 2–3× too high

`PokerEvaluator.identify_hand` returned all five played cards as the scoring set
for Pair, Two Pair, Three of a Kind and Four of a Kind. In Balatro only the cards
that form the hand score; the rest are played and thrown away. So a Pair sitting
inside five cards was advertised with the chips of all five.

Every play from that losing run, estimate against what the game actually paid:

| Play | Old estimate | Game paid | Now |
| --- | ---: | ---: | ---: |
| Pair `AD QD 10D 5S 5C` | ~102 | 40 | 40 |
| Two Pair `QH 9S 9D 8H 8D` | ~128 | 108 | 108 |
| Pair `KH 7D 6D 4H 4C` | ~82 | 36 | 36 |
| Pair `KD 10C 9C 3H 3C` | ~90 | 32 | 32 |

Those four estimates are the whole basis on which the model was choosing. They
also drove the `STRONG`/`DECENT` labels, so the run was told a 40-chip pair
covered 34% of a 300-chip blind when it covered 13%. Locked down in
`tests/test_poker_eval.py`.

## 2. Laya never saw its own cards

Laya's English checkpoint reads 512 tokens, of which the options take up to 192,
leaving ~320 for the state — and the state is truncated **from the end**. It was
being handed the full BalatroLLM markdown gamestate, which spends its first
thousand-odd tokens on phase, jokers, consumables, vouchers and a table of all
nine poker hand levels before it reaches `## Current Hand`.

The hand was past the cut. The model was choosing between descriptions of cards
it could not see.

`sidecar/brief.py` now builds a purpose-made brief of about 100 tokens that puts
the need, the resources and the available hands first. LLM backends still get the
full markdown, which they have room for.

## 3. The instructions were being truncated to about 8 tokens

Laya lays a question out as `[CLS] instructions [SEP] [MASK] opt0 [MASK] opt1 …
[SEP] state`, and charges the options against `head_max_len` **first**:

```python
opt_budget = head_max_len - sum(len(o) for o in opt_ids)
head_ids = head_ids[: max(8, opt_budget)]
```

The old prompts were 60–80 token strategy paragraphs behind up to nine options of
40–50 tokens each. The options alone blew the 192-token head, so the instructions
were clipped to eight tokens and each option was clipped to a few characters.
Silently — no error, no warning.

Questions are now ≤6 options of ≤30 tokens. `backends.head_budget()` measures it
and `tests/test_agent.py::TestPromptBudget` fails if any question the tree can
ask overruns either checkpoint's budget, jokers and synergy tags included.

## 4. `laya_tree` was running the wrong checkpoint

It loaded the English root checkpoint (trained at 512/192) and then overrode its
config to 1024/256 — the budgets belonging to `typed-decisions`, a different
checkpoint. That is the worst of both: positions the root was not trained on, and
none of the fine-tuning that makes typed questions work. On the model's own
benchmark the root scores 0.362 and `typed-decisions` scores 0.766.
`LayaTreeBackend` now loads `typed-decisions` itself and leaves its budgets alone.

## 5. Python was making the decision anyway

The tree tagged options `[BEST]`, `[2nd best]`, `[3rd best]` and then offered only
the top four by Python's own score. A model that can read "BEST" is not choosing
a Balatro move; it is reading a label Python wrote. Alongside it: the runner
bought a booster pack on its own if money ≥ 25, `find_weakest_joker_index`
picked which joker to sell, and every failure path fell back to the
highest-scoring hand — so a broken model still looked like it was playing.

All removed. Options now carry **absolute** labels (`WINS`, `STRONG`, `OK`,
`WEAK`, `TINY`, plus a pace fact like *"3 more like it still lose"*) that describe
one option against the blind and never against its siblings. Fallbacks increment
`backend.fallbacks`, say `FALLBACK` in the reasoning, and are printed by the
runner and the bench. `tests/test_decision_boundary.py` asserts that every option
offered is reachable and that different answers produce different moves.

---

## Measurements

`sidecar/stress.py` plays the offline simulator, so a change can be measured in
seconds instead of a game session. Mean rounds won over 50 seeds, Red-deck rules
(8 cards, 4 hands, 4 discards), no shop:

### No jokers

| Policy | Tree | Flat |
| --- | ---: | ---: |
| `random` | 0.70 | 0.40 |
| `first` (always option 1) | 0.92 | 0.92 |
| `label` (reads only the option text) | **2.28** | 1.62 |

### Three jokers (Joker, Greedy Joker, Banner)

| Policy | Tree | Flat |
| --- | ---: | ---: |
| `first` | 9.68 | 9.68 |
| `label` | **9.80** | 9.76 |

Two things to read out of this:

- **The tree is worth it.** Given the same information, splitting the turn into
  small questions is worth about +40% rounds over one wide question, and the gap
  comes from the discard branch: flat mode buries the discard plans among the
  play options, and the probe stops digging.
- **Joker facts matter more than joker names.** The jokered table first read
  7.36 for `label` — *worse* than `first` — because Banner pays for each unused
  discard, so digging was costing score. Surfacing that as a fact in the brief
  ("Your jokers pay for each unused discard, so discarding lowers your score")
  moved it to 9.80. Nothing about the decision logic changed; the model was
  simply told something true that it had no way to work out.

`label` is not an oracle. It is a fixed rule that reads the same option text the
model reads, so it measures what the labels are worth, not what a perfect player
would score.

What that looks like in play — `--policy label --seeds 1 --verbose`, seed 0:

```
WON  Ante 1 Small Blind: 320/300
    DISCARD KD JS 4H 5D AD · 2C 8C 7C 6C KC · 9S 10H 3S 8H · 10S JH 6S 5H
    PLAY 8D JD 7D QD 10D -> 320      (Flush)
WON  Ante 1 Big Blind: 1120/450
    DISCARD 9S 8S 6H AH QC · JH 10C 4C 7H 3S · KS 8C 7C 8H · 3D 4D 9H 2C
    PLAY 9D 8D 7D 10D 6D -> 1120     (Straight Flush)
WON  Ante 1 Boss Blind: 1120/600
    DISCARD 10C AD 9C 6D · 4D 2D KC 2C
    PLAY 7S 6S 8S 9S 10S -> 1120     (Straight Flush)
```

Four discards spent digging, then one hand that ends the blind — which is how
Balatro is actually played, and the exact behaviour the old tree never produced
because it was told a 40-chip pair covered a third of the target.

---

## Where Laya itself lands

`typed-decisions` checkpoint, 24 seeds, no jokers, mean rounds won:

| | Tree | Flat |
| --- | ---: | ---: |
| `laya` | **1.04** | 0.75 |
| beat the first blind | 79% | 71% |
| cleared Ante 1 | 4% | 0% |

For scale, on the same simulator: `random` 0.70, `first` 0.92, `label` 2.28.

So with the scoring fixed, the hand visible and the questions inside the budget,
zero-shot Laya is clearly above random, roughly level with always taking the
first option, and far below what the labels themselves support. The tree helps
it (1.04 vs 0.75) in the same direction it helps the probe.

### What the model can and cannot answer

`sidecar/qbench.py` asks decision points generated from real dealt hands, in the
tree's own format, where the simulator knows the right answer. Each one is asked
twice: once with the real option text, and once with every number and verdict
stripped out. 360 items, 30 hands, `typed-decisions` checkpoint:

| question | variant | accuracy | chance | same answer |
| --- | --- | ---: | ---: | ---: |
| which of two hands scores more | labelled | **100%** | 50% | 10% |
| which of two hands scores more | bare | **100%** | 50% | 10% |
| which of all of them scores most | labelled | **93%** | 25% | 10% |
| which of all of them scores most | bare | 50% | 25% | 50% |
| can any of them beat the blind | labelled | 47% | 50% | 63% |
| can any of them beat the blind | bare | 50% | 50% | 97% |
| play or dig | labelled | 50% | 50% | **100%** |
| play or dig | bare | 50% | 50% | **100%** |

Read the last column first. On the two yes/no questions the model gives the same
answer to every board -- 100% of the time for play-or-dig. Its accuracy there is
not "wrong", it is "not answering". Asking them as a two-option choice instead of
a native yes/no changes nothing.

And then the top two rows. Asked which of two hands is bigger it is right every
single time, **including with every number and verdict deleted** -- it knows a
Flush beats a Pair from its language prior, without being told what either is
worth. With the full list it is right 93% of the time with labels and at chance
without, so the labels are doing real work once the comparison gets harder.

The split is clean and it is not about arithmetic being hard to *read*: in
"can any of them beat the blind" both numbers are right there in the text. It is
about the kind of question. **Relations between poker hands: excellent.
Quantities: nothing.** That is what a System 1 pattern matcher with a language
prior looks like from the outside, and it is exactly the distinction the tree now
builds on -- the play-or-dig call is asked as "which poker hand do you want to
score with", not as "is this enough".

Moving to that shape took two attempts. Listing hand *types* with one
representative each made things worse than the old two-question tree (0.75
against 1.04 mean rounds over 24 seeds), because a type with a stand-in is not
the comparison the model is good at. Listing the hands themselves gives **1.17**,
the best measured so far.

### What the live game showed: it skips almost every blind

The offline harness has no blind-selection decision -- it plays every blind in
order. The first live bench run made that gap visible immediately: on the five
official seeds Laya **skipped the Big Blind four times out of four**, walking
into the Boss one shop early. Two of those runs died at 474/600 and 460/600,
and both numbers beat the Big Blind's 450 they had just declined to play.

The first fix was to give the question the facts it was missing. The brief named
the tag and said playing pays cash, but never said what came instead: the Boss's
name, its target and its debuff all sit in the gamestate and were not being
shown. They are now, along with "skipping pays no cash and opens no shop".

It made no difference -- the next run skipped *more*, including the Small Blind,
and died in Round 1 on most seeds. That is the signature of a question that is
not being read rather than a question that is missing information, and it lines
up with the poll result below: both are `noul` questions, and both behave as if
the answer barely depends on the text.

So the same decision can now be asked either way (`binary_nodes`), and `qbench`
asks its yes/no items both ways on identical boards, which is what the
`same answer` column in its output is for: a model that answers the same thing
whatever the board scores 100% there, and its accuracy is then only the base rate
of that answer.

### In the actual game

Fifteen seeds -- the five official BalatroBench ones and ten random -- Red deck,
White stake, `fallbacks: 0` throughout, so every move below was the model's:

| | cleared Ante 1 | cleared Ante 2 | best run |
| --- | ---: | ---: | --- |
| before any of this | 0 / 5 | 0 / 5 | died Round 2, 216/300 |
| after the scoring and context fixes | 1 / 15 | 1 / 15 | Ante 3 |
| after reshaping the questions | **4 / 15 (27%)** | 1 / 15 | Ante 3, Round 7 |

The blind question moved with it: from 33 skips against 1 play, to 29 against 11.
It is still the weakest node -- the model answers it at around 0.07 confidence,
which is a coin flip -- but it is no longer skipping every blind it is offered
and walking into the Boss a shop early.

So a 421M encoder that has never seen Balatro, doing no arithmetic and never
generating a token, clears the first Ante about a quarter of the time and has
cleared the second. That is a long way from Ante 8 and a long way from nothing.

### Boss blinds: the facts were simply wrong

The offline harness plays ordinary blinds, so two bugs survived until the live
bench reached bosses.

A face-down card -- The Mark, The Wheel, The House, The Fish -- reports no rank
and no suit, and `parse_state` defaulted it to the 2 of Spades. Every hand the
model was offered on those blinds could contain a card that does not exist.
Face-down cards are now excluded from every enumerated hand and discard plan,
shown as `??`, and counted in the brief.

A debuffed card -- The Club, The Plant -- counted its full chips. It now counts
none while still forming the hand type, which is what the game does.

And the boss's own rule, sitting in the gamestate unused, is now the first line
of the brief.

None of this is strategy; all of it is the difference between describing the
board and describing a different board.

### Does the tree actually help the model, or only the probe?

Both, and it helps the model more. Same code, same 24 seeds, no jokers:

| | tree | flat | gain |
| --- | ---: | ---: | ---: |
| `laya` | **1.17** | 0.71 | **+65%** |
| `label` probe | 2.28 | 1.62 | +41% |

The probe reads a wide option list as accurately as a short one, so flat costs it
only the discard plans getting buried. The model loses accuracy as options
multiply, so it gains from the split twice over.

### Where the ceiling is, and why a bigger tree does not move it

Worth being blunt about, because "make the tree better" has a limit that is
already measured.

The `label` probe never miscalculates and never misreads. In the shop-less
simulator it averages 2.28 rounds with no jokers and ~9.8 with three good ones --
Ante 3 or 4. That is the ceiling of *reading these facts perfectly*, and Ante 8
is well above it. What is missing at that point is not better reading: it is
choosing which jokers to buy, when to level hands and when to hold money, across
24 rounds that interact. Nothing in this design looks further ahead than the
decision in front of it.

Nor does throughput help. Laya answers in ~33 ms on a GPU, so thousands of
decisions per run are affordable -- but there is no search to spend them on. Each
call is independent and the model has no notion of lookahead, so asking the same
question a thousand times returns the same answer a thousand times. A search
*would* pay if there were a reliable evaluator to guide it, and qbench says there
is not one for the questions that matter.

The tempting move from here is to encode the plan in Python -- "by Ante 3 you
want an X-Mult joker and level-5 hands" -- and let the model tick off each step.
That would raise the score and destroy the point: the model would be rubber-
stamping a strategy Python wrote, and the run would measure Python.

`--compare` exists to catch exactly that. It reports the policy's distance from
`random` alongside its score; if a change keeps the score up while that gap
collapses, the tree has started playing.

### The routing idea, and why the measurement killed it

The plan was: the one class of question the model answers well -- which hand
scores more, 93-100% on qbench -- is also the most frequent decision in a run, so
let it absorb the volume while an LLM makes the hard calls. Same score, far fewer
API calls.

The flaw is in what qbench was measuring. Its version of that question is "which
of these scores more", and **Python computed those chip counts in order to write
the labels**. A right answer there is agreement with arithmetic Python already
did, not judgement. So the benchmark that justified routing tested something
easier than the thing being routed.

The `greedy` probe settles it: always play the biggest hand on offer, never dig.
Two lines, no model. Over 60 seeds, tree mode, no jokers:

| policy | mean rounds won |
| --- | ---: |
| `random` | 0.53 ± 0.09 |
| **`greedy`** -- plain Python `max()` | **0.93 ± 0.09** |
| `first` | 0.95 ± 0.10 |
| **`laya`** | **1.07** |
| `label` -- reads the labels and digs | 2.30 ± 0.11 |

The model is about four standard errors clear of random, so it is certainly doing
something. It is about one standard error clear of the rule it was supposed to
justify replacing, which is to say not clear of it at all. (At 24 seeds the same
comparison read 1.17 against 1.00; the larger sample pulled it down, which is the
usual direction for a number one was hoping would hold.)

So the whole gap that matters at this level -- 0.93 to 2.30 -- is the play-or-dig
call, which is a question about a quantity, which is exactly what this model does
not answer.

Routing the hand node to the cheap model would therefore hand it the decision
that needs judgement while saving the calls that were nearly free. `HybridBackend`
still exists and is tested, but its trusted list is **empty by default**: no
question type has earned a place on it yet. Name one explicitly and measure the
split before believing it.

There is also no pruning niche to fall back on. Eight cards make 56 five-card
hands and Python scores all of them exactly in under a millisecond, so a cheap
approximate scorer has nothing to approximate, and a rules check on an LLM's
proposed move is something Python does exactly and for free.

### Node shape: one choice, or a poll of yes/no questions?

`noul` is Laya's strongest primitive (0.857 against 0.733 for `choice` on its own
benchmark) and a two-option question can never overrun the head budget, however
many candidates there are. So `rank_options` was added: one yes/no per option
("The player should play WEAK - Pair 9S 9D…"), all in a single batched forward
pass, and the option Laya scores highest wins. Python compares nothing of its
own — every number is the model's.

Measured, it was worse: 0.67 against 1.25 for a single choice question over the
same 12 seeds, and it costs one sequence per option instead of one for the
question. The plausible reason is that `noul` asks whether a statement holds in
isolation, so independently-scored options are not calibrated against each other,
whereas `choice` softmaxes them together. `--node choice` is therefore the
default and `--node poll` stays available; if the model is ever fine-tuned on
this workflow the comparison is worth repeating.

That is consistent with what the model card says out loud: the base checkpoints
score 0.362 on typed decisions against a 0.318 random baseline and a 0.461
majority-class baseline, and the 0.766 belongs to a checkpoint fine-tuned on that
benchmark's own training split. *"Laya is a fast base to specialise, not a
zero-shot decision engine."*

So the next step is not more prompt wording. It is fine-tuning on Balatro
decisions, and the repository now produces exactly that data:

```bash
PYTHONPATH=. .venv/bin/python sidecar/stress.py \
    --policy label --mode tree --seeds 500 --max-ante 8 \
    --jokers joker,greedy,banner --dump data/decisions.jsonl
```

Each row is a question, its options, the answer given, and whether that round was
won — state, action and reward, which is the shape Laya was trained with (RLCD).
