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

---

## Where Laya itself lands

See the table in the README. The short version: with the scoring fixed, the hand
visible and the questions inside the budget, zero-shot Laya moves off the random
floor but stays far below what the labels support.

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
