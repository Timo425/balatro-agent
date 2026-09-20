# balatro-agent — working rules

This repository exists to **test decision models on Balatro**. It is not a bot
that plays Balatro. If the Python here ever gets good at Balatro, the project has
failed at its only job.

Read this before changing anything in `sidecar/`.

---

## The one rule: Python never decides

> **Python enumerates legal moves and states facts about them.
> The model chooses which one happens.**

Every in-game action must be traceable to an answer the backend gave. There is no
"sensible default", no "just this once", no "obviously you play the flush".

### Facts vs decisions

Python is *allowed* — in fact required — to compute anything the game's own rules
determine. It is never allowed to express a preference.

| Python may (facts) | Python may not (decisions) |
| --- | --- |
| Identify that five cards make a Flush | Decide the Flush is the move |
| Compute that the Flush scores ~340 chips | Sort options by score and offer the top 3 |
| Say the blind needs 300 and 340 ≥ 300 → `WINS` | Play it because it wins |
| List every pair the hand can make | Pick "the best pair" |
| Note that a joker rewards Hearts | Discard the non-Hearts |
| State that $5 of $12 is "affordable" | Buy because it is affordable |
| Report which jokers are eternal (unsellable) | Choose which sellable joker goes |

The test: *if the fact changed, would the move change on its own?* If yes, Python
is deciding.

### Specific things that are forbidden

- Ranking labels across options — `[BEST]`, `[2nd best]`, `top-3`, `★★★★` scales
  that compare siblings. Per-option absolute labels (`WINS`, `WEAK`) are fine.
- `sorted(options, key=score)[:n]` before showing them to the model. Cutting the
  list is choosing.
- `if score >= target: return play(...)` — any early return that skips the model.
- Heuristic fallbacks that happen to play well (e.g. "buy a pack if money ≥ 25").
- Using a model score to *veto* the model's own answer.

### Ordering and tie-breaks

Options are presented in the game's own fixed order (poker-hand ranking, then
rank order). That ordering is a property of the rules and does not change with
the situation, so it carries no hint. Two mechanical tie-breaks are allowed and
are documented where they happen:

- When two candidates make the same hand type and one strictly scores more with
  no other difference, only the higher is enumerated (`candidates.py`).
- If the model says "send filler cards", the filler is the cheapest spare cards
  (`pick_filler`). *Whether* to send filler is the model's call.

### Fallbacks

A fallback exists only so a dead backend cannot crash the run. When one fires it
must:

1. increment `backend.fallbacks`,
2. put `FALLBACK` in the action's `reasoning`,
3. surface in the runner/bench output.

A run with `fallbacks > 0` is Python's play, not the model's, and must not be
reported as a model result.

---

## Where the boundary lives in the code

| File | Role |
| --- | --- |
| `sidecar/poker_eval.py` | Hand identification and which cards score. Rules only. |
| `sidecar/scoring.py` | Score estimates and absolute labels (`WINS`/`STRONG`/…). Facts only. |
| `sidecar/candidates.py` | Enumerates legal plays and named discard plans. Never ranks. |
| `sidecar/synergy.py` | Reads joker effect text into "what this rewards". Annotates, never picks. |
| `sidecar/brief.py` | The compact board description the model reads. |
| `sidecar/agent.py` | Asks the questions, executes the answers. **The decision boundary.** |
| `sidecar/backends.py` | The model behind the questions (`laya`, `laya_tree`, `api_llm`, `agentic`). |
| `sidecar/sim.py` | Offline simulator. Executes moves it is handed; contains no strategy. |
| `sidecar/stress.py` | Probe policies for measurement. **Never import this into the play path.** |

`tests/test_decision_boundary.py` enforces this: every option the agent offers
must be reachable, different answers must produce different moves, a broken
backend must fall back visibly, and no play-path module may import `stress`.

If you add a decision, add it to that test.

---

## Writing questions the decision model can actually answer

Laya is a 421M non-autoregressive encoder. It scores each option at a marker
token in a single forward pass. Consequences that are easy to get wrong:

- **It cannot do arithmetic.** Any number it needs to compare must already be
  resolved in the text (`~108 chips, WEAK, 3 more like it still lose`).
- **Options and instructions share one budget** (`head_max_len`: 192 on the
  English root, 256 on `typed-decisions`). Options are charged first; when they
  overrun, the *instructions* are cut to as few as 8 tokens and each option is
  clipped — silently. Keep to ≤6 options of ≤30 tokens. `backends.head_budget()`
  measures it and `tests/test_agent.py::TestPromptBudget` guards it.
- **State is truncated from the end**, so the decision-relevant facts go first.
  This is why `brief.py` exists: the full BalatroLLM markdown gamestate is many
  times the budget, and feeding it to Laya cut the player's own hand out of every
  prompt.
- **The option key is part of the text** the model reads, so keys are meaningful
  words (`flush_hearts`, `keep_pairs`), not `item_3`.
- **Binary is its strongest primitive** (`noul` 0.857 vs `choice` 0.733 on its
  own benchmark), and a two-option question never overruns the head budget.
  Hence `predict_bool` and the `rank_options` poll node.

---

## Measuring before believing

```bash
# Offline: no game needed, thousands of rounds in seconds
PYTHONPATH=. .venv/bin/python sidecar/stress.py --policy label --mode both --seeds 50

# The real model (slow: ~0.3 s per decision on CPU)
USE_TF=0 PYTHONPATH=. .venv/bin/python sidecar/stress.py --policy laya --mode tree --seeds 12
```

`--policy label` is a fixed rule reading only the option text the model sees. It
is not an oracle, so a low score does not prove the tree is at fault -- but it
does say what the labels alone are worth. If it does well and the model does not,
the gap is the model's; if neither does, look at the labels first.

`--dump PATH` writes every question, its options, the answer and whether the
round was won, as JSONL: state, action and reward, ready for a fine-tune.

Run the suite with `./.venv/bin/python -m unittest discover -s tests`.
Set `BALATRO_TEST_LAYA=1` to include the tests that load the real checkpoint.
`docs/findings.md` records what was wrong before this design and the numbers
behind it.

---

## Conventions

- Code, comments and docs in this repo are in English (it builds on English
  upstreams and the docs are read by models). Terminal output is partly Estonian.
- The game install stays vanilla; play on Profile 2/3, never Profile 1.
