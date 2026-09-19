import os
import json
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/media/timo/Storage/.cache"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import laya
from sidecar.poker_eval import PokerEvaluator, Card, EvaluatedHand

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

        # Rule 1: If top hand beats the blind, ALWAYS play immediately
        if top_hand.total_score >= needed_chips:
            return AgentAction(
                action_type="PLAY",
                cards=top_hand.cards,
                reasoning=f"Winning play: '{top_hand.hand_type}' yields {top_hand.total_score} pts, beating remaining {needed_chips} pts.",
                confidence=1.0
            )

        # Rule 2: Flush Digging Heuristic (from Section 2 of strategy rulebook)
        # If holding 3 or 4 cards of a single suit, and current hand is weak, dig for the Flush!
        if state.discards_left > 0:
            suit_counts = Counter(c.suit for c in state.hand)
            most_common_suit, count = suit_counts.most_common(1)[0]
            if 3 <= count < 5 and top_hand.total_score < (needed_chips * 0.5):
                discards = [c for c in state.hand if c.suit != most_common_suit][:5]
                return AgentAction(
                    action_type="DISCARD",
                    cards=discards,
                    reasoning=f"Strategy Rule 2 (Flush Digging): Holding {count} {most_common_suit}. Discarding {len(discards)} off-suit cards to dig for Flush.",
                    confidence=0.98
                )

        # Rule 3: For ambiguous situations where discards remain, query Laya
        if state.discards_left > 0 and state.hands_left > 1:
            recommended_discards = PokerEvaluator.recommend_discards(state.hand)
            laya_state = {
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
                    "instructions": "Determine if current hand score justifies playing or if player should discard.",
                    "criteria": {
                        "play": "The hand scores high enough or remaining hands are critical.",
                        "discard": "The hand score is too low compared to target chips and discards are plentiful."
                    }
                }
            }

            res = self.agent.predict(laya_state, questions)
            choice = res["answers"]["hand_strategy"]["choice"]
            conf = float(res["answers"]["hand_strategy"].get("confidence", 0.90))

            if choice == "discard":
                return AgentAction(
                    action_type="DISCARD",
                    cards=recommended_discards,
                    reasoning=f"Neural decision: Discarding {len(recommended_discards)} cards to improve hand score.",
                    confidence=conf
                )

        # Default fallback: Play the best hand
        return AgentAction(
            action_type="PLAY",
            cards=top_hand.cards,
            reasoning=f"Playing best available hand: {top_hand.hand_type} ({top_hand.total_score} pts).",
            confidence=0.95
        )

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

        # Strategy Keywords for high-priority early Jokers
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
                        return (one_based_idx, f"Buying '{item.get('name')}' () to establish Mult/Chip engine.")
                else:
                    # Mid/Late game: check if high tier or keeping interest cap (5)
                    is_elite = any(k in name for k in scaling_keywords)
                    if is_elite or (state.money - cost >= 20):
                        return (one_based_idx, f"Acquiring scaling/utility Joker '{item.get('name')}' ().")

        # If multiple affordable items and still deciding, consult Laya
        if affordable and can_hold_joker and state.money >= 6:
            cand_summary = [f"#{idx}: {item.get('name')} ()" for idx, item, cost in affordable]
            laya_state = {
                "ante": state.ante,
                "money": state.money,
                "jokers": state.jokers,
                "available_items": cand_summary
            }
            questions = {
                "shop_decision": {
                    "type": "choice",
                    "instructions": "Choose whether to buy an available card or save money for interest.",
                    "criteria": {
                        "buy_first": "Buy the first available item to strengthen deck.",
                        "save": "Save money towards 5 interest threshold."
                    }
                }
            }
            try:
                res = self.agent.predict(laya_state, questions)
                choice = res["answers"]["shop_decision"]["choice"]
                if choice == "buy_first":
                    first_idx, first_item, first_cost = affordable[0]
                    return (first_idx, f"Laya recommended purchasing '{first_item.get('name')}' ().")
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
