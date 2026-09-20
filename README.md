# balatro-agent

A test bench for **decision models playing Balatro**.

The point of this repository is to find out how far a given model can get when it
is the only thing making decisions. Python deals the information; the model plays
the game. There is no built-in bot, and there is deliberately no fallback that is
good at Balatro — see [CLAUDE.md](CLAUDE.md) for the rule and how it is enforced.

Three kinds of player can sit behind the same interface:

| Backend | What it is | Notes |
| --- | --- | --- |
| `laya` | [Laya](https://huggingface.co/convaiinnovations/laya), a 421M non-autoregressive decision model, one wide question per decision | The flat baseline |
| `laya_tree` | The same model driving a hierarchical decision tree | Uses the `typed-decisions` checkpoint |
| `api_llm` / `api_llm_tree` | A frontier LLM over an OpenAI-compatible API | For benchmarking |
| `agentic` | An external agent loop or a human at the console | How an agentic LLM plays turn by turn |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Balatro (Proton / Steam) + coder/balatrobot mod          │
│   JSON-RPC 2.0 on 127.0.0.1:12346                        │
└──────────────────────┬───────────────────────────────────┘
                       │ gamestate  ▲ play / discard / buy
┌──────────────────────▼───────────────────────────────────┐
│ FACTS (Python)                                           │
│   poker_eval.py   which cards make what, and which score │
│   scoring.py      chips a play is worth, absolute labels │
│   synergy.py      what each joker rewards                │
│   candidates.py   every legal play and named discard plan│
│   brief.py        a board description small enough to fit│
├──────────────────────────────────────────────────────────┤
│ DECISIONS (the model)                                    │
│   agent.py        asks the questions, executes answers   │
│   backends.py     laya · laya_tree · api_llm · agentic   │
└──────────────────────────────────────────────────────────┘
```

---

## The decision tree

One wide "pick a move" question is the wrong shape for a small decision model:
its options share a fixed token budget, so the more there are, the less of each
one it can read. The tree splits the turn into a few small questions instead.

```
SELECTING_HAND
├── dig_for_better_hand?           yes/no    throw cards away, or play now
│   ├── yes → which_discard        ≤6 named plans
│   │         keep the 4 Hearts (1 short of a Flush) · keep the 9s and 8s ·
│   │         keep the 4 cards in a row · keep the Spades your jokers reward
│   └── no  → which_hand           ≤6 options, each labelled with what it is worth
│             "Two Pair 9S 9D 8H 8D: ~108 chips, OK, needs 3 hands like it"
│             ↳ past 6 options: which_hand_type, then which_hand_of_type
├── send_filler_cards?             yes/no    cycle the spare cards or keep them
SHOP
├── shop_category                  joker · pack · voucher · consumable · save
├── which_shop_item                with synergy notes against your current jokers
└── sell_joker_for_room            only when slots are full
BLIND_SELECT
└── skip_blind?                    yes/no
```

Nothing in that tree is pre-filtered or ranked by Python. Every option carries an
**absolute** label (`WINS`, `STRONG`, `OK`, `WEAK`, `TINY`) describing how it
stands against the blind on its own — never how it compares with its siblings.

## What the model is told about jokers

Joker text is parsed into what it rewards, and that shows up wherever it is
relevant:

- in the score of every candidate play (`Banner +90 chips`, `Cavendish x3 mult`),
- as a tag on options that hit the synergy (`🃏 3× Hearts → Lusty Joker`),
- as a discard plan (`keep the Hearts your jokers reward`),
- as a shop note (`reinforces your Hearts build` / `adds Flush synergy`),
- and as a board-level consequence: *"Your jokers pay for each unused discard, so
  discarding lowers your score."*

Effects that cannot be read off their text (retriggers, scaling counters,
random chances) are reported as `plus unread joker text` rather than silently
dropped, so the number is honestly a floor.

---

## Running it

```bash
# Play a live game (Balatro must be running with the mod, or use --auto-launch)
PYTHONPATH=. .venv/bin/python sidecar/runner.py

# BalatroBench: 5 official seeds, Red deck / White stake
PYTHONPATH=. .venv/bin/python sidecar/bench.py --backend laya_tree --auto-launch

# Offline stress test: thousands of rounds, no game needed
PYTHONPATH=. .venv/bin/python sidecar/stress.py --policy label --mode both --seeds 50
PYTHONPATH=. .venv/bin/python sidecar/stress.py --policy laya --mode tree --seeds 12

# Tests
./.venv/bin/python -m unittest discover -s tests
BALATRO_TEST_LAYA=1 ./.venv/bin/python -m unittest discover -s tests   # + real model
```

### The stress probes

`sidecar/stress.py` plays the offline simulator with scripted answerers, so a
change to the tree can be measured in seconds:

- `random` — the floor.
- `first` — always the first option, which is what Python's own fallback does.
- `label` — a fixed rule reading *only* the option text the model sees. It is not
  an oracle, but it shows what the labels alone are worth.
- `laya` — the real model.

`fallbacks` in the output must be `0`. Anything else means Python answered for the
model and the run does not measure the model.

---

## Safety & profile separation

- The official game install stays **100% vanilla**.
- Switch to **Profile 2 or 3** in Balatro before running the agent; Profile 1
  keeps your personal, unmodded progress.

---

## Component provenance

| Component | Upstream | Contribution |
| --- | --- | --- |
| Game mod / IPC bridge | [`coder/balatrobot`](https://github.com/coder/balatrobot) v1.5.2 | JSON-RPC 2.0 API for reading state and dispatching actions |
| Strategy & prompts | [`coder/balatrollm`](https://github.com/coder/balatrollm) | `strategies/default/` templates and `rules/balatro-strategy.md` |
| Decision model | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large System 1 decision model, ~0.3 s per decision on CPU |
| Everything else | this repository | Fact extraction, the decision tree, the offline simulator and the stress harness |
