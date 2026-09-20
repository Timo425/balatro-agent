"""
Balatro Autonomous AI Agent
Driven 100% by Laya (ModernBERT System 1) decisions using BalatroLLM strategy templates.
Python acts solely as the interface: enumerating legal candidate actions with real
in-game scores and executing Laya's decisions via JSON-RPC.
"""

import itertools
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sidecar.poker_eval import Card, PokerEvaluator
from sidecar.strategy import StrategyManager
from sidecar.backends import BaseBackend, create_backend

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

    def __init__(
        self,
        rules_path: str = "rules/balatro-strategy.md",
        strategy_name: str = "default",
        backend: Any = "laya",
        model: Optional[str] = None,
        tree_mode: bool = False,
        **backend_kwargs,
    ):
        self.rules_path = Path(rules_path)
        self.strategy_name = strategy_name
        self.sm = StrategyManager(strategy_name)
        if isinstance(backend, BaseBackend):
            self.backend = backend
        else:
            self.backend = create_backend(backend_type=str(backend), model=model, **backend_kwargs)
        self.agent = getattr(self.backend, "agent", None)
        self.tree_mode = (
            tree_mode
            or getattr(self.backend, "is_tree", False)
            or str(backend).lower() in ("laya_tree", "tree", "hierarchical")
        )

    def _get_max_chars(self) -> int:
        return getattr(self.backend, "max_chars", 2000)

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

        if self.tree_mode:
            return self._decide_hand_action_tree(state, raw_g)


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
            chosen_key = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "hand_action",
                (
                    "You are playing Balatro. Strategic rules: "
                    "1. Always prioritize playing winning 5-card hands (Flush, Full House, Straight) if they beat or heavily damage the target chips. "
                    "2. If current hands are weak (high card or low pair) and target chips are high, prioritize discarding off-suit or low cards to dig for a Flush or synergy. "
                    "3. Select the single best strategic action."
                ),
                criteria,
            )
            if chosen_key in candidates:
                act_type, cards, desc = candidates[chosen_key]
                return AgentAction(
                    action_type=act_type,
                    cards=cards,
                    reasoning=f"Decision: {desc}",
                    confidence=0.90,
                )
        except Exception:
            pass

        first_key = next(iter(candidates))
        act_type, cards, desc = candidates[first_key]
        return AgentAction(act_type, cards, f"Defaulting to {desc}", 0.8)

    def _decide_hand_action_tree(self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None) -> AgentAction:
        """Hierarchical decision tree:
        Stage 1: Intent (PLAY vs DISCARD).
        Stage 2A: Hand selection (if PLAY).
        Stage 2B: Discard selection (if DISCARD).
        """
        hands_data = raw_g.get("hands", {}) if raw_g else {}
        needed_chips = max(0, state.target_chips - state.current_chips)

        # 1. Enumerate playable multi-card hands
        play_candidates: Dict[str, Tuple[List[Card], str, int]] = {}
        seen_hand_types = set()
        for size in (5, 4, 3, 2):
            for combo in itertools.combinations(state.hand, size):
                h_type, _ = PokerEvaluator.identify_hand(list(combo))
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
                    play_candidates[key] = (list(combo), desc, est_pts)

        # High card options
        sorted_by_val = sorted(state.hand, key=lambda c: c.value)
        highest_card = sorted_by_val[-1]
        h_info = hands_data.get("High Card", {})
        lvl = h_info.get("level", 1)
        base_c = h_info.get("chips", 5)
        base_m = h_info.get("mult", 1)
        hc_pts = (base_c + highest_card.chip_value) * base_m
        play_candidates["play_high_card"] = (
            [highest_card],
            f"Play High Card [{highest_card.rank}{highest_card.suit[0]}] (Level {lvl}: ~{hc_pts} pts)",
            hc_pts,
        )

        if len(state.hand) > 1 and state.discards_left == 0:
            lowest_card = sorted_by_val[0]
            play_candidates["play_single_low"] = (
                [lowest_card],
                f"Play single card [{lowest_card.rank}{lowest_card.suit[0]}] as High Card to preserve remaining cards",
                base_c + lowest_card.chip_value,
            )

        # 2. Enumerate discard options
        discard_candidates: Dict[str, Tuple[List[Card], str]] = {}
        if state.discards_left > 0:
            suit_counts = Counter(c.suit for c in state.hand)
            most_common_suit, count = suit_counts.most_common(1)[0]
            if 3 <= count < 5:
                offsuits = [c for c in state.hand if c.suit != most_common_suit][:5]
                off_str = ", ".join(f"{c.rank}{c.suit[0]}" for c in offsuits)
                discard_candidates["discard_offsuit"] = (
                    offsuits,
                    f"Discard {len(offsuits)} non-{most_common_suit} cards [{off_str}] to dig for the 5th {most_common_suit} for an 800+ pt Flush",
                )

            num_discard = min(len(state.hand), 4)
            low_discard = sorted_by_val[:num_discard]
            low_str = ", ".join(f"{c.rank}{c.suit[0]}" for c in low_discard)
            discard_candidates["discard_lowest"] = (
                low_discard,
                f"Discard {len(low_discard)} lowest cards [{low_str}] to dig for stronger hands or synergies",
            )

        best_play_key = max(play_candidates, key=lambda k: play_candidates[k][2])
        best_play_cards, best_play_desc, best_play_pts = play_candidates[best_play_key]
        has_winning_hand = best_play_pts >= needed_chips

        # Stage 1: Macro Intent (PLAY vs DISCARD)
        # Always query Laya directly whenever discards are available — no Python heuristic overrides.
        chosen_intent = "intent_play"
        if state.discards_left > 0 and discard_candidates:
            intent_criteria = {
                "intent_play": f"PLAY: Play a hand now (Best available hand: {best_play_desc})",
                "intent_discard": f"DISCARD: Discard cards to dig through deck for better cards (Discards left: {state.discards_left})",
            }
            macro_context = (
                f"# Round Status: Ante {state.ante} ({state.blind_name})\n"
                f"- Needed chips: {needed_chips} | Current: {state.current_chips} / {state.target_chips}\n"
                f"- Hands left: {state.hands_left} | Discards left: {state.discards_left}\n"
                f"- Best available hand: {best_play_desc}\n"
            )
            try:
                chosen_intent = self.backend.predict_choice(
                    macro_context,
                    "macro_intent",
                    (
                        "Strategic rule: If you hold a winning hand or strong combo, choose PLAY. "
                        "If current hands are weak and discards are available, choose DISCARD to dig through the deck."
                    ),
                    intent_criteria,
                )
            except Exception:
                chosen_intent = "intent_play"



        # Stage 2A: Play
        if chosen_intent != "intent_discard":
            if len(play_candidates) == 1:
                return AgentAction("PLAY", best_play_cards, f"Tree Decision: {best_play_desc}", 0.95)

            sorted_plays = sorted(play_candidates.items(), key=lambda kv: kv[1][2], reverse=True)
            top_plays = dict(sorted_plays[:4])
            play_criteria = {k: desc for k, (_, desc, _) in top_plays.items()}
            gamestate_text = self._render_state(raw_g, state)

            try:
                chosen_play = self.backend.predict_choice(
                    gamestate_text[:self._get_max_chars()],
                    "play_hand",
                    "Choose the single best poker hand to play from the candidate options to maximize scoring and beat the blind.",
                    play_criteria,
                )
                if chosen_play in top_plays:
                    cards, desc, _ = top_plays[chosen_play]
                    return AgentAction("PLAY", cards, f"Tree Decision (Play): {desc}", 0.95)
            except Exception:
                pass
            return AgentAction("PLAY", best_play_cards, f"Tree Fallback: {best_play_desc}", 0.90)

        # Stage 2B: Discard
        else:
            if len(discard_candidates) == 1:
                k = next(iter(discard_candidates))
                cards, desc = discard_candidates[k]
                return AgentAction("DISCARD", cards, f"Tree Decision: {desc}", 0.95)

            discard_criteria = {k: desc for k, (_, desc) in discard_candidates.items()}
            gamestate_text = self._render_state(raw_g, state)
            try:
                chosen_discard = self.backend.predict_choice(
                    gamestate_text[:self._get_max_chars()],
                    "discard_hand",
                    "Choose which cards to discard to best improve the remaining hand.",
                    discard_criteria,
                )
                if chosen_discard in discard_candidates:
                    cards, desc = discard_candidates[chosen_discard]
                    return AgentAction("DISCARD", cards, f"Tree Decision (Discard): {desc}", 0.95)
            except Exception:
                pass
            first_k = next(iter(discard_candidates))
            cards, desc = discard_candidates[first_k]
            return AgentAction("DISCARD", cards, f"Tree Fallback: {desc}", 0.90)

    def _decide_shop_action_tree(
        self,
        state: BalatroGameState,
        shop_items: List[Dict[str, Any]],
        max_jokers: int = 5,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, int, str]]:
        """Hierarchical decision tree for shop actions:
        Stage 1: Category Intent (Buy Joker vs Buy Pack vs Buy Voucher vs Save Money).
        Stage 2: Item Selection within chosen category.
        Stage 3: Slot Management (if buying joker and slots are full).
        """
        if state.money < 1 or not shop_items:
            return None

        val_j = raw_g.get("jokers") if raw_g else None
        if isinstance(val_j, dict):
            current_jokers = val_j.get("cards", [])
        elif isinstance(val_j, list):
            current_jokers = val_j
        else:
            current_jokers = []
        can_hold_joker = len(current_jokers) < max_jokers

        joker_items = []
        pack_items = []
        voucher_items = []
        other_items = []

        for idx, item in enumerate(shop_items):
            cost = item.get("cost", 4)
            if cost > state.money:
                continue
            name = (item.get("name") or item.get("label") or "").lower()
            set_type = (item.get("set") or "").upper()
            if "voucher" in name or set_type == "VOUCHER":
                voucher_items.append((idx, item))
            elif "pack" in name or set_type == "BOOSTER":
                pack_items.append((idx, item))
            elif any(w in name for w in ["planet", "tarot", "spectral"]) or set_type in ("PLANET", "TAROT", "SPECTRAL"):
                other_items.append((idx, item))
            else:
                joker_items.append((idx, item))

        has_sellable = any(
            isinstance(j, dict) and not (isinstance(j.get("modifier"), dict) and j.get("modifier").get("eternal"))
            for j in current_jokers
        )
        eligible_for_joker = can_hold_joker or (has_sellable and joker_items)

        # Stage 1: Categories
        categories = {}
        if joker_items and eligible_for_joker:
            categories["cat_joker"] = f"Buy a Joker (Available: {len(joker_items)} affordable jokers, Current slots: {len(current_jokers)}/{max_jokers})"
        if pack_items:
            categories["cat_pack"] = f"Buy a Booster Pack (Available: {len(pack_items)} affordable packs)"
        if voucher_items:
            categories["cat_voucher"] = f"Buy a permanent Voucher upgrade (Available: {len(voucher_items)})"
        if other_items:
            categories["cat_consumable"] = f"Buy Consumable item (Available: {len(other_items)})"
        categories["cat_save"] = f"End shop and save money (Bank: ${state.money}, earns $1 interest per $5 held up to $25)"

        if len(categories) <= 1:
            return None

        cat_context = (
            f"# Shop Decisions - Ante {state.ante}\n"
            f"- Money: ${state.money}\n"
            f"- Joker slots: {len(current_jokers)}/{max_jokers}\n"
            f"- Current jokers: {', '.join(j.get('label') or j.get('key', '') for j in current_jokers if isinstance(j, dict))}\n"
        )

        chosen_cat = "cat_save"
        try:
            chosen_cat = self.backend.predict_choice(
                cat_context,
                "shop_category",
                (
                    "You are at the Balatro shop. Decide whether to purchase a Joker, Booster Pack, Voucher, or save money towards interest. "
                    "Prioritize building powerful Joker synergies in early-mid game. Save money if offerings are weak."
                ),
                categories,
            )
        except Exception:
            return None

        if chosen_cat == "cat_save":
            return None

        # Stage 2: Item Selection within chosen category
        target_list = []
        if chosen_cat == "cat_joker":
            target_list = joker_items
        elif chosen_cat == "cat_pack":
            target_list = pack_items
        elif chosen_cat == "cat_voucher":
            target_list = voucher_items
        else:
            target_list = other_items

        if not target_list:
            return None

        chosen_idx, chosen_item = target_list[0]
        if len(target_list) > 1:
            item_criteria = {}
            for idx, it in target_list:
                it_name = it.get("name") or it.get("label") or "Item"
                it_cost = it.get("cost", 4)
                val_obj = it.get("value") if isinstance(it.get("value"), dict) else {}
                it_eff = it.get("effect") or val_obj.get("effect") or ""
                eff_desc = f" ({it_eff})" if it_eff else ""
                item_criteria[f"item_{idx}"] = f"Buy '{it_name}' (${it_cost}){eff_desc}"

            gamestate_text = self._render_state(raw_g, state)
            try:
                chosen_key = self.backend.predict_choice(
                    gamestate_text[:self._get_max_chars()],
                    "select_shop_item",
                    "Select the best item in this category to purchase for your deck.",
                    item_criteria,
                )
                if chosen_key.startswith("item_"):
                    c_idx = int(chosen_key.split("_")[1])
                    for idx, it in target_list:
                        if idx == c_idx:
                            chosen_idx, chosen_item = idx, it
                            break
            except Exception:
                pass

        # Stage 3: Slot Replacement if buying a Joker with full 5/5 slots
        # Query Laya directly to choose which existing joker to sell — no Python heuristics.
        if chosen_cat == "cat_joker" and not can_hold_joker:
            sell_criteria = {}
            for idx, j in enumerate(current_jokers):
                if not isinstance(j, dict):
                    continue
                mod = j.get("modifier")
                if isinstance(mod, dict) and mod.get("eternal"):
                    continue
                j_name = j.get("label") or j.get("key", f"Joker #{idx}")
                val_obj = j.get("value") if isinstance(j.get("value"), dict) else {}
                j_eff = j.get("effect") or val_obj.get("effect") or ""
                eff_desc = f" ({j_eff})" if j_eff else ""
                cost_obj = j.get("cost")
                sell_val = cost_obj.get("sell", 1) if isinstance(cost_obj, dict) else (cost_obj if isinstance(cost_obj, int) else 1)
                sell_criteria[f"sell_{idx}"] = f"Sell '{j_name}' (+${sell_val}){eff_desc} to free a slot for '{chosen_item.get('name', 'Joker')}'"

            sell_criteria["cancel_buy"] = f"Do not sell any joker (cancel purchasing '{chosen_item.get('name', 'Joker')}')"

            gamestate_text = self._render_state(raw_g, state)
            try:
                chosen_sell = self.backend.predict_choice(
                    gamestate_text[:self._get_max_chars()],
                    "sell_joker_to_make_room",
                    f"Your joker slots are full (5/5). Choose which existing joker to sell to make room for '{chosen_item.get('name', 'Joker')}', or cancel.",
                    sell_criteria,
                )
                if chosen_sell.startswith("sell_"):
                    s_idx = int(chosen_sell.split("_")[1])
                    s_joker = current_jokers[s_idx]
                    s_name = s_joker.get("label") or s_joker.get("key", f"Joker #{s_idx}")
                    return ("SELL", s_idx, f"Tree Decision: Sell '{s_name}' to free slot for '{chosen_item.get('name', 'Joker')}'.")
            except Exception:
                pass
            return None


        return ("BUY", chosen_idx, f"Tree Decision: Buy '{chosen_item.get('name', 'Item')}' from shop.")

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

        if self.tree_mode:
            return self._decide_shop_action_tree(state, shop_items, max_jokers, raw_g)


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
                if can_hold_joker or any(w in name.lower() for w in ["voucher", "planet", "tarot", "pack", "reroll"]):
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
            choice = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "shop_action",
                (
                    "You are playing Balatro. Review available shop items, money, and current jokers. "
                    "If a powerful scaling/scoring joker is available, buy it (or sell a weak joker if slots are full). "
                    "Otherwise save money towards $25 interest."
                ),
                criteria,
            )
            if choice in candidates and choice != "end_shop":
                act_type, target_idx, desc = candidates[choice]
                return (act_type, target_idx, f"Decision: {desc}")
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
            res = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "blind_action",
                (
                    "Decide whether to play this blind for round payout and shop access, or skip for the tag reward. "
                    "Only skip if tag offers game-changing value (Negative, Polychrome, Free shop items)."
                ),
                candidates,
            )
            if res == "skip_blind":
                return ("SKIP", f"Skipping {blind_name} for Tag: {tag_name}")
            return ("SELECT", f"Playing {blind_name} for cash and shop.")
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
            choice = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "pack_choice",
                "Evaluate the booster pack offerings and select the card that best improves the current deck and jokers.",
                candidates,
            )
            if choice.startswith("card_"):
                chosen_idx = int(choice.split("_")[1])
                chosen_card = pack_cards[chosen_idx]
                card_name = chosen_card.get("label") or chosen_card.get("key", "")
                return (chosen_idx, None, f"Decision: Selected '{card_name}' from pack.")
        except Exception:
            pass

        return (0, None, f"Selected '{pack_cards[0].get('label', 'card')}'")

    @staticmethod
    def consumable_requires_hand(c: Dict[str, Any]) -> bool:
        """Determines if a consumable card targets cards in hand (requiring SELECTING_HAND state)."""
        set_type = (c.get("set") or "").upper()
        if set_type == "PLANET":
            return False

        name = (c.get("label") or c.get("key") or "").lower()
        non_hand_names = {
            "hermit", "the hermit", "fool", "the fool", "temperance",
            "judgement", "the judgement", "wheel of fortune", "the wheel of fortune",
            "emperor", "the emperor", "high priestess", "the high priestess"
        }
        if any(nh in name for nh in non_hand_names):
            return False

        val_obj = c.get("value", {}) if isinstance(c, dict) else {}
        eff = (val_obj.get("effect") or "").lower() if isinstance(val_obj, dict) else ""
        if eff:
            if re.search(r"select(?:ed)?\s+(\d+\s+)?cards?|\bconvert\b|\bdestroy\b|\benhance\b", eff):
                return True
            if any(k in eff for k in ["money", "cash", "joker", "copy", "edition", "creates up to"]):
                return False

        return set_type == "TAROT"

    def decide_tarot_targets(
        self,
        tarot_name: str,
        hand: Optional[List[Any]] = None,
        raw_g: Optional[Dict[str, Any]] = None,
        effect: str = "",
    ) -> Optional[List[int]]:
        """Queries Laya directly to choose which card(s) in hand to target with a Tarot card.
        Determines target requirements dynamically from card effect text without hardcoded name lists.
        """
        if not hand:
            return None

        eff_lower = (effect or "").lower()
        t_name_lower = tarot_name.lower()

        # Check if this card targets hand cards based on effect text or card description
        # In Balatro, targeting cards state "selected card" or "selected cards"
        if eff_lower:
            target_match = re.search(r"(?:select(?:ed)?\s+(?:up\s+to\s+)?(\d+)\s+cards?|(\d+)\s+(?:up\s+to\s+)?(?:selected\s+)?cards?)", eff_lower)
            if target_match:
                max_targets = int(target_match.group(1) or target_match.group(2))
            elif "selected card" in eff_lower or "select 1 card" in eff_lower:
                max_targets = 1
            elif any(k in eff_lower for k in ["convert", "destroy", "enhance", "enhances"]):
                max_targets = 2 if ("2" in eff_lower or "two" in eff_lower) else 1
            else:
                return None

        else:
            # If no effect text was provided, detect self-targeting tarots (money/joker tarots)
            # which never target playing cards in hand
            non_targeting_keywords = ["money", "cash", "joker", "copy", "edition"]
            if any(k in t_name_lower for k in non_targeting_keywords):
                return None
            max_targets = 1

        candidates = {}
        for idx, c in enumerate(hand):
            r = getattr(c, "rank", "?")
            s = getattr(c, "suit", "?")
            candidates[f"card_{idx}"] = f"Target {r} of {s} (index {idx})"

        if not candidates:
            return None

        gamestate_text = self._render_state(raw_g)
        questions = {
            "tarot_target": {
                "type": "choice",
                "instructions": (
                    f"You are applying Tarot card '{tarot_name}'. "
                    "Select the single best card in hand to enhance, convert, or destroy based on your strategy."
                ),
                "criteria": candidates,
            }
        }

        try:
            choice = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "tarot_target",
                f"You are applying Tarot card '{tarot_name}'. Select the single best card in hand to enhance, convert, or destroy based on your strategy.",
                candidates,
            )
            if choice.startswith("card_"):
                chosen_idx = int(choice.split("_")[1])
                if max_targets >= 2 and len(hand) >= 2:
                    if self.tree_mode:
                        remaining_candidates = {
                            k: desc
                            for k, desc in candidates.items()
                            if k != f"card_{chosen_idx}"
                        }
                        if remaining_candidates:
                            try:
                                choice2 = self.backend.predict_choice(
                                    gamestate_text[:self._get_max_chars()],
                                    "tarot_target_2",
                                    f"Select the second card in hand to target with '{tarot_name}'.",
                                    remaining_candidates,
                                )
                                if choice2.startswith("card_"):
                                    c2 = int(choice2.split("_")[1])
                                    return [chosen_idx, c2]
                            except Exception:
                                pass
                    c1 = 1 if chosen_idx != 1 else 0
                    return [chosen_idx, c1]
                return [chosen_idx]
        except Exception:
            pass


        if max_targets >= 2 and len(hand) >= 2:
            return [0, 1]
        return [0]

    def decide_consumable_action(
        self,
        consumables: List[Dict[str, Any]],
        hand: Optional[List[Any]] = None,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, Optional[List[int]], str]]:
        """Presents consumables in inventory to Laya to decide if and when to use them."""
        if not consumables:
            return None

        current_state = (raw_g.get("state") or "") if raw_g else ""
        is_selecting_hand = (current_state == "SELECTING_HAND")

        candidates = {}
        for idx, c in enumerate(consumables):
            if not is_selecting_hand and self.consumable_requires_hand(c):
                continue
            name = c.get("label") or c.get("key", f"Consumable #{idx}")
            set_type = c.get("set", "CONSUMABLE")
            val_obj = c.get("value", {}) if isinstance(c, dict) else {}
            eff = val_obj.get("effect", "") if isinstance(val_obj, dict) else ""
            eff_desc = f" ({eff})" if eff else ""
            candidates[f"use_{idx}"] = f"Use {set_type} '{name}'{eff_desc} immediately to level up hand or gain bonus"

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
            choice = self.backend.predict_choice(
                gamestate_text[:self._get_max_chars()],
                "consumable_action",
                "Determine whether to use an available consumable card now or hold it in inventory. Using planets or money tarot immediately is usually optimal.",
                candidates,
            )
            if choice.startswith("use_"):
                idx = int(choice.split("_")[1])
                name = consumables[idx].get("label") or consumables[idx].get("key", "")
                val_obj = consumables[idx].get("value", {}) if isinstance(consumables[idx], dict) else {}
                eff = val_obj.get("effect", "") if isinstance(val_obj, dict) else ""
                targets = self.decide_tarot_targets(name, hand=hand, raw_g=raw_g, effect=eff)
                return (idx, targets, f"Decision: {candidates[choice]} (targets={targets})")
        except Exception:
            pass

        return None

    def decide_joker_order(self, current_jokers: List[Dict[str, Any]], raw_g: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        """Presents joker reordering options directly to Laya to decide the optimal trigger sequence.
        Evaluates joker effects dynamically (phase 0: Chips/Utility, phase 1: Flat Mult, phase 2: X-Mult)
        without any hardcoded joker names.
        """
        if len(current_jokers) <= 1:
            return None

        indexed = []
        for i, j in enumerate(current_jokers):
            val_obj = j.get("value", {}) if isinstance(j, dict) else {}
            effect = (val_obj.get("effect") or j.get("effect") or "").lower() if isinstance(val_obj, dict) else ""

            # Classify scoring phase purely from effect mechanics:
            # Multiplicative mult (X-Mult): "X3", "x1.5", "X Mult" -> triggers last
            if re.search(r"\bx\s*\d+(\.\d+)?|\bx\s*mult\b|\bxmult\b", effect):
                cat = 2
            # Flat mult (+Mult): "+10 Mult", "gives +3 Mult" -> triggers middle
            elif re.search(r"\b\+\s*\d+\s*mult\b|\bmult\b", effect):
                cat = 1
            # Chips or utility -> triggers first
            else:
                cat = 0
            indexed.append((cat, i))

        candidate_order = [idx for cat, idx in sorted(indexed, key=lambda x: x[0])]
        if candidate_order == list(range(len(current_jokers))):
            return None

        curr_names = [current_jokers[i].get("label") or current_jokers[i].get("key", f"Joker #{i}") for i in range(len(current_jokers))]
        new_names = [current_jokers[i].get("label") or current_jokers[i].get("key", f"Joker #{i}") for i in candidate_order]

        prompt_lines = [
            "Balatro evaluates Jokers sequentially from left to right during scoring.",
            "Rule: Chips and +Mult must trigger first, and X-Mult (multiplicative Mult) must trigger last on the far right to maximize score.",
            "Current jokers:"
        ]
        for i, j in enumerate(current_jokers):
            lbl = j.get("label") or j.get("key", f"Joker #{i}")
            eff = (j.get("value", {}).get("effect") if isinstance(j.get("value"), dict) else j.get("effect")) or ""
            prompt_lines.append(f"Position {i}: {lbl}" + (f" ({eff})" if eff else ""))

        prompt_text = "\n".join(prompt_lines)

        candidates = {
            "order_original": f"Keep original order: {chr(44).join(curr_names)}",
            "order_optimized": f"Order: {chr(44).join(new_names)} (positions +Chips/+Mult first, and X-Mult on far right to maximize score)",
        }

        questions = {
            "order": {
                "type": "choice",
                "instructions": "Select the optimal left-to-right arrangement of Jokers to maximize final score.",
                "criteria": candidates,
            }
        }

        try:
            choice = self.backend.predict_choice(
                prompt_text,
                "order",
                "Select the optimal left-to-right arrangement of Jokers to maximize final score.",
                candidates,
            )
            if choice == "order_optimized":
                return candidate_order
            elif choice == "order_original":
                return None
        except Exception:
            pass

        return candidate_order
