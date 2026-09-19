# Balatro Strategy & Living Rulebook (Complete Operational Reference)

This operational rulebook governs the autonomous Balatro agent. It unifies combinatorial poker evaluations, game engine endpoints, and Laya's neural decision engine into strict, actionable policies.

---

## RULE 1: Hand Evaluation & Play Priority

1. **Instant Win Rule (RULE_INSTANT_WIN):**
   - If the highest-scoring immediately playable hand beats or matches remaining blind chips, **PLAY IMMEDIATELY**.
   - Never waste discards when the round is already mathematically secured.

2. **Flush Digging Heuristic (RULE_FLUSH_DIG):**
   - If holding $\ge 3$ cards of a single suit and current best hand achieves $< 50\%$ of remaining blind chips, discard up to 5 non-matching cards to dig for the Flush.

3. **Hand Type Tier Hierarchy:**
   - **Tier 0:** Straight Flush, Four of a Kind (game-winning high multipliers).
   - **Tier 1:** Flush, Full House (reliable primary scoring engines for Ante 1–4).
   - **Tier 2:** Straight, Three of a Kind (situational fallbacks).
   - **Tier 3:** Two Pair (good with flat mult support).
   - **Tier 4:** Pair, High Card (only viable with heavy scaling or dedicated Jokers).

4. **Boss Blind Counters (RULE_BOSS_COUNTERS):**
   - **The Psychic (Must play 5 cards):** Always pad hands with junk cards to exactly 5 cards.
   - **The Eye (No repeat hand types):** Never play the same poker hand type consecutively.
   - **The Arm (Decreases hand level):** Focus on one-shotting the blind with your highest-level hand.
   - **The Needle (Play only 1 hand):** Spend all discards to craft the absolute strongest 5-card hand.

---

## RULE 2: Joker Architecture & Positioning

Balatro triggers Jokers sequentially from **LEFT to RIGHT**. Positioning directly dictates exponential scoring.

### 4 Joker Pillars:
1. **+Chips (Left):** Establishes chip foundation (e.g., Blue Joker, Banner, Ice Cream, Bull, Stuntman).
2. **+Flat Mult (Middle-Left):** Additive mult before multiplying (e.g., Gros Michel, Half Joker, Popcorn, Mystic, Abstract).
3. **+Scaling Mult (Middle-Right):** Permanent growth engines (e.g., Ride the Bus, Green Joker, Supernova, Red Card).
4. **X-Mult (Far Right):** Exponential multipliers (e.g., Cavendish, Card Sharp, Constellation, Blackboard, Blueprint, Brainstorm).

### Ordering Policy (RULE_JOKER_ORDER):
- **ALWAYS arrange:** `[+Chips] -> [+Flat Mult] -> [+Scaling Mult] -> [X-Mult]`.
- Trigger `rearrange` endpoint immediately whenever a new Joker is purchased or acquired.

---

## RULE 3: Shop & Economy Management

1. **The $25 Interest Threshold (RULE_INTEREST_RESERVE):**
   - Balatro awards $1 interest per $5 held, capped at $5 interest for $25.
   - **Ante 1–2 (Survival Phase):** Aggressively buy flat mult Jokers and chips to survive. Do not hoard if under lethal risk.
   - **Ante 3+ (Compound Interest Phase):** Maintain $\ge \$25$ cash reserve whenever possible. Spend only the surplus above $25 unless a tier-1 Joker or pack appears.

2. **Purchase Priority in Shop (RULE_SHOP_PRIORITY):**
   - **Priority 1 (Survival/Scaling Jokers):** Fill empty Joker slots with flat mult (early) or X-mult / scaling (mid/late).
   - **Priority 2 (Vouchers):** Buy economy vouchers (+1 hand, +1 discard, extra shop slot, interest bonus).
   - **Priority 3 (Booster Packs when $\ge \$25$):**
     - *Buffoon Pack:* If open Joker slots or seeking X-Mult upgrades.
     - *Celestial Pack:* Always buy to upgrade Flush (Jupiter) or high-frequency hands.
     - *Arcana Pack:* Money (Hermit), suit conversion, or card enhancements.
   - **Priority 4 (Reroll):** Never reroll below $25 unless under lethal threat in the upcoming blind.

---

## RULE 4: Consumables (Planet & Tarot) Usage Policy

1. **Planet Cards (RULE_USE_PLANET):**
   - Planet cards (Jupiter, Saturn, Earth, Mars, etc.) must be consumed **IMMEDIATELY** via the `use` endpoint to permanently level up base chips and mult. Never let Planet cards clog consumable slots.

2. **Tarot Cards (RULE_USE_TAROT):**
   - *The Hermit (Doubles money up to $20):* Use immediately if holding $\ge \$10$ or if slot space is needed.
   - *The Fool:* Use immediately to duplicate the last valuable Planet or Tarot card.
   - *Suit Converters (Moon, Sun, Star, World):* Use on off-suit cards to build a single-suit Flush deck.
   - *Enhancements (Empress, Hierophant, Magician):* Apply to Aces and high court cards.
