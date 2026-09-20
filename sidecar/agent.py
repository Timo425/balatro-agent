"""
Balatro Autonomous AI Agent
Driven 100% by Laya (ModernBERT System 1) decisions using BalatroLLM strategy templates.
Python acts solely as the interface: enumerating legal candidate actions with real
in-game scores and executing Laya's decisions via JSON-RPC.
"""

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import laya
from sidecar.poker_eval import Card, PokerEvaluator
from sidecar.strategy import StrategyManager

DEFAULT_HAND_SCORES = {
    "Straight Flush": (100, 8),
    "Four of a Kind": (60, 7),
    "Full House": (40, 4),
    "Flush": (35, 4),
    "Straight": (30, 4),
    "Three of a Kind": (30, 3),
    "Two Pair": (20, 2),
    "Pair": (10, 2),
    "High Card": (5, 1),
}


@dataclass
class BalatroGameState:
    ante: int
    blind_name: str
    target_chips: int
    current_chips: int
    hands_left: int
    discards_left: int
    money: int
    hand: List[Card]
    jokers: List[str]


@dataclass
class AgentAction:
    action_type: str  # "PLAY" or "DISCARD"
    cards: List[Card]
    reasoning: str
    confidence: float


def find_weakest_joker_index(jokers: List[Dict[str, Any]]) -> Optional[int]:
    """Finds the lowest-tier sellable joker when slots are full."""
    if not jokers:
        return None
    for idx, j in enumerate(jokers):
        if not isinstance(j, dict):
            continue
        mod = j.get("modifier")
        if isinstance(mod, dict) and mod.get("eternal"):
            continue
        return idx
    return 0


class BalatroAgent:
    """Autonomous Balatro agent where Laya makes 100% of game decisions."""

    def __init__(self, rules_path: str = "rules/balatro-strategy.md", strategy_name: str = "default"):
        self.rules_path = Path(rules_path)
        self.strategy_name = strategy_name
        self.sm = StrategyManager(strategy_name)
        self.agent = laya.load("convaiinnovations/laya")

    def _render_state(self, raw_g: Optional[Dict[str, Any]], state: Optional[BalatroGameState] = None) -> str:
        """Renders gamestate markdown via BalatroLLM Jinja template."""
        if raw_g:
            try:
                return self.sm.render_gamestate(raw_g)
            except Exception:
                pass
        if state:
            return (
                f"# Balatro Game State\n"
                f"- Ante: {state.ante} | Blind: {state.blind_name}\n"
                f"- Target: {state.target_chips} | Current: {state.current_chips}\n"
                f"- Hands Left: {state.hands_left} | Discards Left: {state.discards_left}\n"
                f"- Money: ${state.money}\n"
                f"- Jokers: {', '.join(state.jokers) if state.jokers else 'None'}\n"
            )
        return "# Balatro Game State (in progress)"

    def decide_hand_action(self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None) -> AgentAction:
        """
        Enumerates all distinct playable hands (using live in-game chip/mult levels from raw_g)
        and discard options, then queries Laya to choose the action.
        """
        if not state.hand:
            return AgentAction("PLAY", [], "No cards in hand", 0.0)

        hands_data = raw_g.get("hands", {}) if raw_g else {}
        needed_chips = max(0, state.target_chips - state.current_chips)

        candidates = {}  # action_key -> (action_type, cards, description)

        # 1. Enumerate distinct multi-card poker hands
        seen_hand_types = set()
        for size in (5, 4, 3, 2):
            for combo in itertools.combinations(state.hand, size):
                h_type, scoring = PokerEvaluator.identify_hand(list(combo))
                if h_type not in seen_hand_types and h_type != "High Card":
                    seen_hand_types.add(h_type)
                    h_info = hands_data.get(h_type, {})
                    lvl = h_info.get("level", 1)
                    default_c, default_m = DEFAULT_HAND_SCORES.get(h_type, (10, 2))
                    base_c = h_info.get("chips", default_c)
                    base_m = h_info.get("mult", default_m)
                    card_chips = sum(c.chip_value for c in combo)
                    est_pts = (base_c + card_chips) * base_m
                    cards_str = ", ".join(f"{c.rank}{c.suit[0]}" for c in combo)
                    key = f"play_{h_type.lower().replace(' ', '_')}"
                    status = "WINNING PLAY!" if est_pts >= needed_chips else f"yields ~{est_pts} pts"
                    desc = f"Play {h_type} [{cards_str}] (Level {lvl}: {status}, needed: {needed_chips})"
                    candidates[key] = ("PLAY", list(combo), desc)

        # 2. Add High Card (1 card: highest rank or lowest rank)
        sorted_by_val = sorted(state.hand, key=lambda c: c.value)
        highest_card = sorted_by_val[-1]
        h_info = hands_data.get("High Card", {})
        lvl = h_info.get("level", 1)
        base_c = h_info.get("chips", 5)
        base_m = h_info.get("mult", 1)
        hc_pts = (base_c + highest_card.chip_value) * base_m
        candidates["play_high_card"] = (
            "PLAY",
            [highest_card],
            f"Play High Card [{highest_card.rank}{highest_card.suit[0]}] (Level {lvl}: ~{hc_pts} pts)",
        )

        if len(state.hand) > 1 and state.discards_left == 0:
            lowest_card = sorted_by_val[0]
            candidates["play_single_low"] = (
                "PLAY",
                [lowest_card],
                f"Play single card [{lowest_card.rank}{lowest_card.suit[0]}] as High Card to preserve remaining cards",
            )

        # 3. Enumerate discard options if discards are available
        if state.discards_left > 0:
            # A. Discard off-suit cards if 3+ of same suit (Flush digging)
            suit_counts = Counter(c.suit for c in state.hand)
            most_common_suit, count = suit_counts.most_common(1)[0]
            if 3 <= count < 5:
                offsuits = [c for c in state.hand if c.suit != most_common_suit][:5]
                off_str = ", ".join(f"{c.rank}{c.suit[0]}" for c in offsuits)
                candidates["discard_offsuit"] = (
                    "DISCARD",
                    offsuits,
                    f"Discard {len(offsuits)} non-{most_common_suit} cards [{off_str}] to dig for the 5th {most_common_suit} for an 800+ pt Flush",
                )

            # B. Discard lowest unaligned cards
            num_discard = min(len(state.hand), 4)
            low_discard = sorted_by_val[:num_discard]
            low_str = ", ".join(f"{c.rank}{c.suit[0]}" for c in low_discard)
            candidates["discard_lowest"] = (
                "DISCARD",
                low_discard,
                f"Discard {len(low_discard)} lowest cards [{low_str}] to dig for stronger hands or synergies",
            )

        # Build Laya criteria
        criteria = {k: desc for k, (_, _, desc) in candidates.items()}
        gamestate_text = self._render_state(raw_g, state)

        questions = {
            "hand_action": {
                "type": "choice",
                "instructions": (
                    "You are playing Balatro. Strategic rules: "
                    "1. Always prioritize playing winning 5-card hands (Flush, Full House, Straight) if they beat or heavily damage the target chips. "
                    "2. If current hands are weak (high card or low pair) and target chips are high, prioritize discarding off-suit or low cards to dig for a Flush or synergy. "
                    "3. Select the single best strategic action."
                ),
                "criteria": criteria,
            }
        }

        try:
            res = self.agent.predict(gamestate_text[:2000], questions)
            chosen_key = res["answers"]["hand_action"]["choice"]
            conf = float(res["answers"]["hand_action"].get("confidence", 0.90))
            if chosen_key in candidates:
                act_type, cards, desc = candidates[chosen_key]
                return AgentAction(
                    action_type=act_type,
                    cards=cards,
                    reasoning=f"Laya Decision: {desc}",
                    confidence=conf,
                )
        except Exception:
            pass

        first_key = next(iter(candidates))
        act_type, cards, desc = candidates[first_key]
        return AgentAction(act_type, cards, f"Defaulting to {desc}", 0.8)

    def decide_shop_action(
        self,
        state: BalatroGameState,
        shop_items: List[Dict[str, Any]],
        max_jokers: int = 5,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, int, str]]:
        """
        Presents all available shop purchases and joker sales to Laya.
        Laya directly decides what to buy, sell, or whether to save money.
        """
        if state.money < 1:
            return None

        candidates = {}  # action_key -> (action_type, index, description)

        val_j = raw_g.get("jokers") if raw_g else None
        if isinstance(val_j, dict):
            current_jokers = val_j.get("cards", [])
        elif isinstance(val_j, list):
            current_jokers = val_j
        else:
            current_jokers = []
        can_hold_joker = len(current_jokers) < max_jokers

        # 1. Affordable items
        for idx, item in enumerate(shop_items):
            cost = item.get("cost", 4)
            if cost <= state.money:
                name = item.get("name", "Item")
                if can_hold_joker or "voucher" in name.lower() or "planet" in name.lower() or "tarot" in name.lower():
                    candidates[f"buy_item_{idx}"] = (
                        "BUY",
                        idx,
                        f"Buy '{name}' (${cost}) from shop",
                    )

        # 2. Joker sales if slots are full
        if not can_hold_joker and current_jokers:
            for idx, j in enumerate(current_jokers):
                if not isinstance(j, dict):
                    continue
                mod = j.get("modifier")
                if isinstance(mod, dict) and mod.get("eternal"):
                    continue
                j_name = j.get("label") or j.get("key", f"Joker #{idx}")
                cost_obj = j.get("cost")
                sell_val = cost_obj.get("sell", 1) if isinstance(cost_obj, dict) else (cost_obj if isinstance(cost_obj, int) else 1)
                candidates[f"sell_joker_{idx}"] = (
                    "SELL",
                    idx,
                    f"Sell '{j_name}' to free a slot (+${sell_val}) for elite shop offerings",
                )

        # 3. Option to leave shop and earn interest
        candidates["end_shop"] = (
            "END",
            -1,
            f"End shop without buying (preserve ${state.money} bankroll for interest)",
        )

        criteria = {k: desc for k, (_, _, desc) in candidates.items()}
        gamestate_text = self._render_state(raw_g, state)

        questions = {
            "shop_action": {
                "type": "choice",
                "instructions": (
                    "You are playing Balatro. Review available shop items, money, and current jokers. "
                    "If a powerful scaling/scoring joker is available, buy it (or sell a weak joker if slots are full). "
                    "Otherwise save money towards $25 interest."
                ),
                "criteria": criteria,
            }
        }

        try:
            res = self.agent.predict(gamestate_text[:2000], questions)
            choice = res["answers"]["shop_action"]["choice"]
            if choice in candidates and choice != "end_shop":
                act_type, target_idx, desc = candidates[choice]
                return (act_type, target_idx, f"Laya Decision: {desc}")
        except Exception:
            pass

        return None

    def decide_blind_action(self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Presents blind play vs skip options to Laya. Laya selects whether to play or skip.
        """
        if not raw_g:
            return ("SELECT", "Playing blind.")

        blinds = raw_g.get("blinds", {})
        current_blind_info = None
        current_type = None

        for b_key in ("small", "big", "boss"):
            b_info = blinds.get(b_key)
            if not isinstance(b_info, dict):
                continue
            st = str(b_info.get("status", "")).upper()
            if st in ("SELECT", "SELECTING", "CURRENT", "ACTIVE") or b_info.get("selected"):
                current_blind_info = b_info
                current_type = b_key.upper()
                break

        if not current_blind_info:
            for b_key in ("small", "big", "boss"):
                b_info = blinds.get(b_key)
                if not isinstance(b_info, dict):
                    continue
                st = str(b_info.get("status", "")).upper()
                if st not in ("DEFEATED", "SKIPPED"):
                    current_blind_info = b_info
                    current_type = b_key.upper()
                    break

        if not current_blind_info:
            return ("SELECT", "Playing blind (blind info unavailable).")

        current_type = current_type or "SMALL"

        if current_type == "BOSS":
            boss_name = current_blind_info.get("name") or "Boss Blind"
            return ("SELECT", f"Playing {boss_name} (Boss blinds cannot be skipped).")

        blind_name = current_blind_info.get("name") or f"{current_type.capitalize()} Blind"
        tag_name = current_blind_info.get("tag_name") or current_blind_info.get("tag") or "Tag"
        tag_desc = current_blind_info.get("tag_effect") or ""

        candidates = {
            "play_blind": f"Play {blind_name} to earn cash payout and visit shop",
            "skip_blind": f"Skip {blind_name} for Tag bonus: '{tag_name}' ({tag_desc})",
        }

        gamestate_text = self._render_state(raw_g, state)
        questions = {
            "blind_action": {
                "type": "choice",
                "instructions": (
                    "Decide whether to play this blind for round payout and shop access, or skip for the tag reward. "
                    "Only skip if tag offers game-changing value (Negative, Polychrome, Free shop items)."
                ),
                "criteria": candidates,
            }
        }

        try:
            res = self.agent.predict(gamestate_text[:2000], questions)
            choice = res["answers"]["blind_action"]["choice"]
            if choice == "skip_blind":
                return ("SKIP", f"Laya Decision: Skipping {blind_name} for Tag '{tag_name}' ({tag_desc}).")
        except Exception:
            pass

        return ("SELECT", f"Laya Decision: Playing {blind_name} for round reward and shop access.")

    def decide_pack_card(
        self,
        pack_cards: List[Dict[str, Any]],
        hand: Optional[List[Card]] = None,
        jokers_count: int = 0,
        max_jokers: int = 5,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, Optional[List[int]], str]]:
        """Presents all cards in booster pack to Laya to pick the best card."""
        if not pack_cards:
            return None

        candidates = {}
        for idx, c in enumerate(pack_cards):
            name = c.get("label") or c.get("key", f"Card #{idx}")
            set_type = c.get("set", "CARD")
            candidates[f"card_{idx}"] = f"Select {set_type} '{name}'"

        candidates["skip_pack"] = "Skip booster pack without selecting any card"

        gamestate_text = self._render_state(raw_g)
        questions = {
            "pack_choice": {
                "type": "choice",
                "instructions": "Evaluate the booster pack offerings and select the card that best improves the current deck and jokers.",
                "criteria": candidates,
            }
        }

        try:
            res = self.agent.predict(gamestate_text[:2000], questions)
            choice = res["answers"]["pack_choice"]["choice"]
            if choice.startswith("card_"):
                chosen_idx = int(choice.split("_")[1])
                chosen_card = pack_cards[chosen_idx]
                card_name = chosen_card.get("label") or chosen_card.get("key", "")
                return (chosen_idx, None, f"Laya Decision: Selected '{card_name}' from pack.")
        except Exception:
            pass

        return (0, None, f"Selected '{pack_cards[0].get('label', 'card')}'")

    def decide_consumable_action(
        self,
        consumables: List[Dict[str, Any]],
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, str]]:
        """Presents consumables in inventory to Laya to decide if and when to use them."""
        if not consumables:
            return None

        candidates = {}
        for idx, c in enumerate(consumables):
            name = c.get("label") or c.get("key", f"Item #{idx}")
            set_type = c.get("set", "CONSUMABLE")
            candidates[f"use_{idx}"] = f"Use {set_type} '{name}' immediately to level up hand or gain cash"

        candidates["keep_all"] = "Keep all consumables in inventory for later"

        gamestate_text = self._render_state(raw_g)
        questions = {
            "consumable_action": {
                "type": "choice",
                "instructions": "Determine whether to use an available consumable card now or hold it in inventory. Using planets or money tarot immediately is usually optimal.",
                "criteria": candidates,
            }
        }

        try:
            res = self.agent.predict(gamestate_text[:2000], questions)
            choice = res["answers"]["consumable_action"]["choice"]
            if choice.startswith("use_"):
                idx = int(choice.split("_")[1])
                return (idx, f"Laya Decision: {candidates[choice]}")
        except Exception:
            pass

        return None

    def decide_joker_order(self, current_jokers: List[Dict[str, Any]]) -> Optional[List[int]]:
        """Positions utility/chips (+Chips) and flat Mult (+Mult) before multiplicative Mult (xMult)."""
        if len(current_jokers) <= 1:
            return None

        indexed = []
        for i, j in enumerate(current_jokers):
            name = (j.get("label") or j.get("key", "")).lower()
            if any(k in name for k in ["cavendish", "constellation", "card sharp", "blueprint", "brainstorm", "duo", "trio", "family", "baron"]):
                cat = 2  # xMult at end
            elif any(k in name for k in ["joker", "half", "gros michel", "popcorn", "mystic", "even", "odd", "abstract"]):
                cat = 1  # +Mult in middle
            else:
                cat = 0  # +Chips / utility at front
            indexed.append((cat, i))

        sorted_indices = [idx for cat, idx in sorted(indexed, key=lambda x: x[0])]
        if sorted_indices == list(range(len(current_jokers))):
            return None
        return sorted_indices
