# Balatro Autonomous AI Agent (`balatro-agent`)

An autonomous Balatro playing agent combining:
1. **Living Markdown Strategy Rules** (`rules/balatro-strategy.md`)
2. **Sub-millisecond Combinatorial Poker Evaluator** (`sidecar/poker_eval.py`)
3. **Laya (ModernBERT-large) System 1 Decision Engine** (`sidecar/agent.py`)
4. **Lightweight Lua State Bridge** (`mod/god_view.lua`)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│ Balatro (Proton / Steam)                               │
│   │                                                    │
│   └── Lua Mod (AppData/Roaming/Balatro/Mods/)          │
│         - Extracts G.hand, G.jokers, G.blind to JSON   │
│         - Receives simulated actions (highlight, play) │
└───────┬────────────────────────────────────────────────┘
        │ Local IPC / JSON Bridge
┌───────┴────────────────────────────────────────────────┐
│ balatro-agent (Python Sidecar)                         │
│   │                                                    │
│   ├── 1. PokerEvaluator (56 hand combos in ~0.3 ms)    │
│   ├── 2. Living Strategy Rules (balatro-strategy.md)   │
│   ├── 3. Laya Neural Decision Engine (~20-30 ms CPU)   │
│   └── 4. Action Dispatcher (PLAY / DISCARD / BUY)      │
└────────────────────────────────────────────────────────┘
```

---

## Safety & Profile Separation

- **Core Game Untouched:** The official game install (`steamapps/common/Balatro`) is **100% vanilla and unmodified**.
- **Profile Isolation:** To keep your personal unlocks and achievements safe:
  1. Launch Balatro normally in Steam.
  2. Switch to **Profile 2** or **Profile 3** (e.g. named "AI").
  3. Profile 1 remains your personal, unmodded progress.

---

## Running the Automated Tests

Run the full simulated test suite:
```bash
cd /media/timo/Storage/Repositories/balatro-agent
./.venv/bin/python -m unittest discover -s tests -v
```

### Verified Scenarios:
- **Scenario A (Instant Win):** Identifies winning Flush and plays immediately without wasting discards.
- **Scenario B (Flush Digging):** Discards 4 off-suit cards when holding 4 Hearts against an 800-chip blind.
- **Scenario C (Shop Purchase):** Evaluates shop offerings and chooses flat +Mult in Ante 1.

---

## Component Provenance & Upstream Repositories

If asked how this agent was constructed or which upstream projects it builds upon, here is the exact architectural breakdown and upstream remotes:

| Component | Upstream Source / Remote | Purpose & Contribution |
| :--- | :--- | :--- |
| **Game Mod / IPC Bridge** | [`coder/balatrobot`](https://github.com/coder/balatrobot) (v1.5.2) | Official Balatro Steam mod injecting HTTP JSON-RPC 2.0 API (`http://127.0.0.1:12346`) to read live game state and dispatch actions (`play`, `discard`, `select`, `skip`, `buy`, `sell`, `rearrange`, `use`, `pack`). |
| **Strategy & Prompts** | [`coder/balatrollm`](https://github.com/coder/balatrollm) | Strategy guidelines (`strategies/default/`), Jinja2 state serialization templates (`GAMESTATE.md.jinja`), and the master 42 KB strategy manual (`rules/balatro-strategy.md`). |
| **Neural Decision Engine** | [`agentica-project/Laya`](https://huggingface.co/agentica-project/Laya) | ModernBERT-large lightweight transformer (~20-30 ms CPU inference) evaluating contextual decisions like hand play vs discard and shop purchases. |
| **Autonomous Sidecar** | *This repository* (`balatro-agent`) | Combines the components into a zero-latency automated loop: sub-millisecond poker scoring (`poker_eval.py`), Lua table serialization sanitation (`client.py`), smart joker selling, booster pack target logic, and rich terminal telemetry (`runner.py`). |
