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
