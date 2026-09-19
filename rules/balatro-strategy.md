# Balatro Strategy & Living Rulebook (Complete Reference)

This living reference defines the comprehensive strategy, heuristic scoring models, and decision rubrics for the autonomous Balatro agent. It bridges mathematical hand evaluation and Laya's neural decision engine.

---

## 1. Hand Priority & Scoring Table

Score Formula: `(Base Chips + Card Values + Chip Additions) × (Base Mult + Mult Additions) × (X-Mult Multipliers)`

| Hand Type | Base Chips | Base Mult | Priority Tier | Strategy & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Straight Flush** | 100 | 8 | Tier 0 | Devastating high score; play immediately if formed. |
| **Four of a Kind** | 60 | 7 | Tier 0 | Dominant power hand; scales rapidly with planet cards. |
| **Flush** | 35 | 4 | Tier 1 (Early Game) | Highest consistency in Ante 1; primary strategy for clearing early blinds. |
| **Full House** | 40 | 4 | Tier 1 (Power) | Massive base points; prioritize when natural pairs/trips appear. |
| **Straight** | 30 | 4 | Tier 2 | High score but higher variance; avoid forcing unless open-ended (4-in-a-row). |
| **Three of a Kind** | 30 | 3 | Tier 2 | Solid fallback when 5-card hands are out of reach. |
| **Two Pair** | 20 | 2 | Tier 3 | Dependable floor with flat +Mult Jokers; weak without Joker support. |
| **Pair** | 10 | 2 | Tier 4 | Low base points; only viable with dedicated scaling (e.g. Jolly Joker, Sly Joker). |
| **High Card** | 5 | 1 | Tier 4 (Late Game) | Requires heavy Planet scaling or Baron/Mime endgame builds. |

---

## 2. Discard Heuristics & Turn Policy

1. **The Instant Win Rule:** If the top immediately playable hand scores $\ge$ the remaining blind chips, **PLAY IMMEDIATELY**. Never waste discards when the round is already won.
2. **Flush Digging:** If holding 3 or 4 cards of one suit and the current best hand does not reach 50% of the target blind, discard the 3 to 5 non-matching cards to dig for the Flush.
3. **Emergency Discard Conservation:** Always preserve at least 1 discard unless remaining hands are insufficient to clear the blind.
4. **Boss Blind Counters:**
   - **The Psychic (Must play 5 cards):** Always pad smaller hands (Pairs, High Card) with junk cards up to 5 cards total.
   - **The Eye (No repeat hand types):** Track played hands; alternate between Flush, Two Pair, and Full House.
   - **The Pillar (No repeat cards this Ante):** Discard previously played card ranks.
   - **The Needle (Play only 1 hand):** Spend all discards searching for your highest-scoring single hand.

---

## 3. The 4 Pillars of Joker Construction

Every winning Balatro run requires balancing the four fundamental Joker categories:

1. **+Chips:** Creates the foundational chip base (e.g., Blue Joker, Banner, Ice Cream, Bull, Stuntman).
2. **+Flat Mult:** Required for Ante 1–3 survival (e.g., Gros Michel, Half Joker, Popcorn, Lusty/Greedy/Wrathful Jokers).
3. **+Scaling Mult:** Permanent growth engine for Ante 4+ (e.g., Ride the Bus, Green Joker, Supernova, Red Card).
4. **X-Mult:** Exponential multiplier needed for Ante 6–8 (e.g., Cavendish, Card Sharp, Constellation, Blackboard).

### Critical Joker Ordering Rule:
**Always position +Chips and +Flat Mult to the LEFT, and X-Mult Jokers to the FAR RIGHT.**
*Reason:* Balatro triggers Jokers sequentially from left to right. Adding mult before multiplying maximizes exponential yield: `(Base + Flat) × X-Mult`.

---

## 4. Shop & Economy Strategy

1. **Interest Management:** Every $5 saved yields $1 interest at round end (max $5 interest at $25).
   - In Ante 1–2: Spend only on essential survival Jokers. Build cash reserve toward $25.
   - Once at $\ge \$25$: Spend excess funds above $25 on scaling Jokers, Planet packs, and Tarot packs.
2. **Booster Pack Evaluation:**
   - **Buffoon Pack (Jokers):** High priority if open Joker slots exist and current scoring is below blind requirements.
   - **Celestial Pack (Planets):** Buy to level up your most played hand type (e.g., Jupiter for Flush).
   - **Arcana Pack (Tarot):** Prioritize suit conversion, card enhancement (Foil/Holographic), or instant cash (Hermit).
   - **Spectral Pack:** High risk/reward; avoid if it threatens existing key Jokers.
3. **Reroll Rule:** Never reroll below $25 unless in imminent lethal danger of losing the run.
