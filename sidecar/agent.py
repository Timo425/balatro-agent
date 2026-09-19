import os
import json
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Any, Optional

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/media/timo/Storage/.cache"

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

    def decide_shop_purchase(self, state: BalatroGameState, shop_jokers: List[str]) -> Optional[str]:
        """Uses strategy rubric to select the best Joker in the shop."""
        if not shop_jokers or state.money < 4:
            return None

        laya_state = {
            "current_money": state.money,
            "ante": state.ante,
            "equipped_jokers": state.jokers,
            "available_jokers_in_shop": shop_jokers
        }

        criteria = {
            "buy_flat_mult": "Buy flat +Mult Joker if holding 0 or few mult jokers in early game.",
            "buy_scaling": "Buy scaling +Mult or chip Joker to prepare for mid-game.",
            "buy_xmult": "Buy X-Mult Joker for late-game exponential scaling.",
            "save_money": "Do not buy, save money to reach interest cap ($25)."
        }

        q = {
            "shop_action": {
                "type": "choice",
                "instructions": "Determine the optimal shop strategy based on current Ante and equipped Jokers.",
                "criteria": criteria
            }
        }

        res = self.agent.predict(laya_state, q)
        return res["answers"]["shop_action"]["choice"]
