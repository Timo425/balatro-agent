import laya
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from sidecar.poker_eval import Card, PokerEvaluator, EvaluatedHand

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
    action_type: str  # "PLAY", "DISCARD", "BUY", "SKIP"
    cards: List[Card]
    target_item: Optional[str] = None
    reasoning: str = ""
    confidence: float = 1.0

# 4 Pillars of Jokers classifier for optimal positioning and purchasing
CHIP_JOKERS = {
    "banner", "blue joker", "ice cream", "bull", "stuntman", "hiker",
    "scary face", "arrowhead", "onyx agate", "castle", "odd todd", "sly joker", "crafty joker"
}

FLAT_MULT_JOKERS = {
    "joker", "gros michel", "half joker", "popcorn", "mystic summit", "abstract joker",
    "misprint", "even steven", "droll joker", "clever joker", "wily joker",
    "zodiac joker", "mad joker", "crazy joker", "gluttonous joker", "lusty joker", "wrathful joker", "greedy joker"
}

SCALING_MULT_JOKERS = {
    "ride the bus", "green joker", "supernova", "red card", "swashbuckler",
    "flash card", "fortune teller", "spare trousers", "constellation", "lucky cat", "wee joker"
}

X_MULT_JOKERS = {
    "cavendish", "card sharp", "blueprint", "brainstorm", "blackboard", "baron",
    "duo", "trio", "family", "order", "tribe", "polychrome", "driver's license",
    "the idol", "ancient joker", "bloodstone", "steel joker", "campfire", "ramen"
}

def classify_joker(name: str) -> int:
    """
    Returns priority weight for positioning:
    0: Chips (Left)
    1: Flat Mult (Middle-Left)
    2: Scaling Mult (Middle-Right)
    3: X-Mult (Far Right)
    """
    n = name.lower()
    for k in X_MULT_JOKERS:
        if k in n:
            return 3
    for k in SCALING_MULT_JOKERS:
        if k in n:
            return 2
    for k in FLAT_MULT_JOKERS:
        if k in n:
            return 1
    for k in CHIP_JOKERS:
        if k in n:
            return 0
    return 1  # Default to flat mult position

class BalatroAgent:
    """Autonomous Balatro agent using living strategy rules and Laya neural decisions."""

    def __init__(self, rules_path: str = "rules/balatro-strategy.md"):
        self.rules_path = Path(rules_path)
        self.rules_content = self.rules_path.read_text(encoding="utf-8") if self.rules_path.exists() else ""
        self.agent = laya.load("convaiinnovations/laya")

    def decide_hand_action(self, state: BalatroGameState) -> AgentAction:
        """Decides whether to play the best available hand or discard cards."""
        best_hands = PokerEvaluator.find_best_playable_hands(state.hand, top_k=2)
        top_hand = best_hands[0]
        needed_chips = max(0, state.target_chips - state.current_chips)

        # Rule 1: RULE_INSTANT_WIN
        # If top hand beats the blind, ALWAYS play immediately
        if top_hand.total_score >= needed_chips:
            return AgentAction(
                action_type="PLAY",
                cards=top_hand.cards,
                reasoning=f"RULE_INSTANT_WIN: '{top_hand.hand_type}' yields {top_hand.total_score} pts, beating needed {needed_chips} pts.",
                confidence=1.0
            )

        # Rule 2: RULE_FLUSH_DIG (from Section 1 of strategy rulebook)
        if state.discards_left > 0:
            suit_counts = Counter(c.suit for c in state.hand)
            most_common_suit, count = suit_counts.most_common(1)[0]
            if 3 <= count < 5 and top_hand.total_score < (needed_chips * 0.5):
                discards = [c for c in state.hand if c.suit != most_common_suit][:5]
                return AgentAction(
                    action_type="DISCARD",
                    cards=discards,
                    reasoning=f"RULE_FLUSH_DIG: Holding {count} {most_common_suit}. Discarding {len(discards)} off-suit cards to dig for Flush.",
                    confidence=0.98
                )

        # Rule 3: For ambiguous situations where discards remain, query Laya with rulebook context
        if state.discards_left > 0 and state.hands_left > 1:
            recommended_discards = PokerEvaluator.recommend_discards(state.hand)
            laya_state = {
                "rules": "RULE_HAND_PRIORITY: Conserve hands; discard low unpaired cards if far from blind chips.",
                "ante": state.ante,
                "hands_left": state.hands_left,
                "discards_left": state.discards_left,
                "needed_chips": needed_chips,
                "playable_hand_type": top_hand.hand_type,
                "playable_hand_score": top_hand.total_score,
                "active_jokers": state.jokers
            }

            questions = {
                "hand_strategy": {
                    "type": "choice",
                    "instructions": "Determine if current hand score justifies playing or if player should discard based on rules.",
                    "criteria": {
                        "play": "The hand score is substantial or remaining hands are critical.",
                        "discard": "The hand score is too low compared to target chips and discards are plentiful."
                    }
                }
            }

            try:
                res = self.agent.predict(laya_state, questions)
                choice = res["answers"]["hand_strategy"]["choice"]
                conf = float(res["answers"]["hand_strategy"].get("confidence", 0.90))

                if choice == "discard":
                    return AgentAction(
                        action_type="DISCARD",
                        cards=recommended_discards,
                        reasoning=f"Laya Neural Decision: Discarding {len(recommended_discards)} cards to dig for stronger hand.",
                        confidence=conf
                    )
            except Exception:
                pass

        # Default fallback: Play the best hand
        return AgentAction(
            action_type="PLAY",
            cards=top_hand.cards,
            reasoning=f"Playing best available hand: {top_hand.hand_type} ({top_hand.total_score} pts).",
            confidence=0.95
        )

    def decide_joker_order(self, current_jokers: List[Dict[str, Any]]) -> Optional[List[int]]:
        """
        Implements RULE_JOKER_ORDER:
        Arrange [+Chips] -> [+Flat Mult] -> [+Scaling Mult] -> [X-Mult].
        Returns list of 0-based indices in the new order, or None if already optimal.
        """
        if len(current_jokers) <= 1:
            return None

        # Sort indices by classified category
        indexed = []
        for i, j in enumerate(current_jokers):
            name = j.get("label") or j.get("key", "")
            cat = classify_joker(name)
            indexed.append((cat, i))

        # Stable sort preserving relative order within categories
        sorted_indices = [idx for cat, idx in sorted(indexed, key=lambda x: x[0])]

        # If already optimal, return None
        if sorted_indices == list(range(len(current_jokers))):
            return None

        return sorted_indices

    def decide_consumable_action(self, consumables: List[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
        """
        Implements RULE_USE_PLANET and RULE_USE_TAROT:
        Returns (0-based index, reasoning) if a consumable should be used immediately.
        """
        if not consumables:
            return None

        for idx, c in enumerate(consumables):
            set_type = c.get("set", "").upper()
            name = (c.get("label") or c.get("key", "")).lower()

            # Rule: Planet cards are always consumed immediately to level up hand values
            if set_type == "PLANET" or any(p in name for p in ["jupiter", "pluto", "mars", "saturn", "earth", "mercury", "uranus", "venus", "neptune"]):
                return (idx, f"RULE_USE_PLANET: Using planet card '{c.get('label') or name}' to level up poker hands.")

            # Rule: The Hermit doubles money immediately
            if "hermit" in name:
                return (idx, "RULE_USE_TAROT: Using The Hermit for instant economy boost.")

            # Rule: The Fool duplicates last card
            if "fool" in name:
                return (idx, "RULE_USE_TAROT: Using The Fool.")

        return None

    def decide_pack_card(self, pack_cards: List[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
        """
        Evaluates opened booster pack cards and selects the best one.
        Returns (0-based index, reasoning) or None to skip.
        """
        if not pack_cards:
            return None

        # 1. Look for Planet cards (especially Jupiter for Flush or Saturn for Straight)
        for idx, c in enumerate(pack_cards):
            name = (c.get("label") or c.get("key", "")).lower()
            if "jupiter" in name:
                return (idx, f"Selected high-priority Flush planet 'Jupiter'.")
            if any(p in name for p in ["saturn", "mars", "earth", "uranus"]):
                return (idx, f"Selected planet card '{c.get('label', name)}'.")

        # 2. Look for high-value Jokers or Tarots
        for idx, c in enumerate(pack_cards):
            name = (c.get("label") or c.get("key", "")).lower()
            cat = classify_joker(name)
            if cat in (2, 3):  # Scaling or X-mult
                return (idx, f"Selected premium Joker '{c.get('label', name)}'.")

        # Fallback: take the first card
        return (0, f"Selected '{pack_cards[0].get('label', 'Card')}' from pack.")

    def decide_shop_action(self, state: BalatroGameState, shop_items: List[Dict[str, Any]], max_jokers: int = 5) -> Optional[Tuple[int, str]]:
        """
        Evaluates available shop items against strategy rules and Laya.
        Returns (1-based index, reasoning) if an item should be bought, or None to end shop.
        """
        if not shop_items or state.money < 3:
            return None

        jokers_count = len(state.jokers)
        can_hold_joker = jokers_count < max_jokers

        # Identify affordable items
        affordable = []
        for idx, item in enumerate(shop_items):
            cost = item.get("cost", 4)
            if cost <= state.money:
                affordable.append((idx + 1, item, cost))

        if not affordable:
            return None

        mult_keywords = [
            "joker", "mult", "popcorn", "gros michel", "half", "abstract", 
            "mystic", "misprint", "even steven", "odd todd", "banner", 
            "blue joker", "scary", "droll", "sly", "clever", "wily", 
            "stencil", "supernova", "ride the bus", "green joker", "ceremonial"
        ]
        scaling_keywords = ["cavendish", "constellation", "card sharp", "blueprint", "brainstorm", "trousers", "duo", "trio", "family"]

        for one_based_idx, item, cost in affordable:
            name = str(item.get("name", "")).lower()
            if can_hold_joker:
                # Ante 1-3: aggressively build basic scoring base
                if state.ante <= 3 or jokers_count < 2:
                    is_scoring = any(k in name for k in mult_keywords) or jokers_count == 0
                    if is_scoring:
                        return (one_based_idx, f"RULE_SHOP_PRIORITY: Buying '{item.get('name')}' (${cost}) to establish Mult/Chip foundation.")
                else:
                    # Mid/Late game: check if high tier or keeping interest reserve (RULE_INTEREST_RESERVE)
                    is_elite = any(k in name for k in scaling_keywords)
                    if is_elite or (state.money - cost >= 20):
                        return (one_based_idx, f"RULE_SHOP_PRIORITY: Acquiring scaling/utility Joker '{item.get('name')}' (${cost}).")

        # If multiple affordable items and still deciding, consult Laya with rulebook context
        if affordable and can_hold_joker and state.money >= 6:
            cand_summary = [f"#{idx}: {item.get('name')} (${cost})" for idx, item, cost in affordable]
            laya_state = {
                "rules": "RULE_INTEREST_RESERVE: Maintain $25 for max interest unless buying essential scoring upgrades.",
                "ante": state.ante,
                "money": state.money,
                "jokers": state.jokers,
                "available_items": cand_summary
            }
            questions = {
                "shop_decision": {
                    "type": "choice",
                    "instructions": "Choose whether to buy an available card or save money for interest based on RULE_INTEREST_RESERVE.",
                    "criteria": {
                        "buy_first": "Buy the first available item to strengthen deck.",
                        "save": "Save money towards $25 interest threshold."
                    }
                }
            }
            try:
                res = self.agent.predict(laya_state, questions)
                choice = res["answers"]["shop_decision"]["choice"]
                if choice == "buy_first":
                    first_idx, first_item, first_cost = affordable[0]
                    return (first_idx, f"Laya Decision: Purchasing '{first_item.get('name')}' (${first_cost}) based on rulebook criteria.")
            except Exception:
                pass

        return None

    def decide_shop_purchase(self, state: BalatroGameState, shop_jokers: List[str]) -> Optional[str]:
        """Backward-compatible helper testing strategy rubric."""
        if not shop_jokers or state.money < 4:
            return 'save_money'
        if state.ante <= 3:
            return 'buy_flat_mult'
        return 'buy_scaling'
