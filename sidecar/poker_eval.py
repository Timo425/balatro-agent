import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Card Rank mapping to numerical values
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14
}

# Chip values contributed by raw card ranks in Balatro
RANK_CHIPS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11
}

# Base hand scoring in vanilla Balatro (level 1 hands)
BASE_HAND_SCORES = {
    "Straight Flush": (100, 8),
    "Four of a Kind": (60, 7),
    "Full House": (40, 4),
    "Flush": (35, 4),
    "Straight": (30, 4),
    "Three of a Kind": (30, 3),
    "Two Pair": (20, 2),
    "Pair": (10, 2),
    "High Card": (5, 1)
}

@dataclass
class Card:
    rank: str
    suit: str
    id: int = 0

    @property
    def value(self) -> int:
        return RANK_VALUES.get(self.rank, 0)

    @property
    def chip_value(self) -> int:
        return RANK_CHIPS.get(self.rank, 0)

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

@dataclass
class EvaluatedHand:
    hand_type: str
    cards: List[Card]
    base_chips: int
    base_mult: int
    card_chips: int
    total_score: int

class PokerEvaluator:
    """Fast combinatorial poker evaluator and Balatro score calculator."""

    @staticmethod
    def identify_hand(cards: List[Card]) -> Tuple[str, List[Card]]:
        if not cards:
            return "High Card", []
        """Identifies the poker hand type and returns (hand_name, scoring_cards)."""
        if len(cards) < 5:
            # Sub-5 card evaluation
            ranks = [c.value for c in cards]
            counts = Counter(ranks)
            if any(cnt == 4 for cnt in counts.values()):
                return "Four of a Kind", cards
            elif any(cnt == 3 for cnt in counts.values()):
                return "Three of a Kind", cards
            elif sum(1 for cnt in counts.values() if cnt == 2) == 2:
                return "Two Pair", cards
            elif any(cnt == 2 for cnt in counts.values()):
                return "Pair", cards
            return "High Card", [max(cards, key=lambda c: c.value)]

        ranks = sorted([c.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        is_flush = len(set(suits)) == 1

        # Check straight (including A-2-3-4-5 wheel)
        is_straight = False
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        if len(unique_ranks) >= 5:
            if unique_ranks[0] - unique_ranks[4] == 4:
                is_straight = True
            elif set([14, 5, 4, 3, 2]).issubset(set(unique_ranks)):
                is_straight = True

        counts_list = sorted(rank_counts.values(), reverse=True)

        if is_flush and is_straight:
            return "Straight Flush", cards
        if counts_list[0] == 4:
            return "Four of a Kind", cards
        if counts_list == [3, 2]:
            return "Full House", cards
        if is_flush:
            return "Flush", cards
        if is_straight:
            return "Straight", cards
        if counts_list[0] == 3:
            return "Three of a Kind", cards
        if counts_list == [2, 2, 1]:
            return "Two Pair", cards
        if counts_list[0] == 2:
            return "Pair", cards
        return "High Card", [max(cards, key=lambda c: c.value)]

    @classmethod
    def evaluate_5card_score(cls, cards: List[Card]) -> EvaluatedHand:
        hand_type, scoring_cards = cls.identify_hand(cards)
        base_chips, base_mult = BASE_HAND_SCORES[hand_type]
        card_chips = sum(c.chip_value for c in cards)
        total_chips = base_chips + card_chips
        total_score = total_chips * base_mult
        return EvaluatedHand(
            hand_type=hand_type,
            cards=cards,
            base_chips=base_chips,
            base_mult=base_mult,
            card_chips=card_chips,
            total_score=total_score
        )

    @classmethod
    def find_best_playable_hands(cls, full_hand: List[Card], top_k: int = 3) -> List[EvaluatedHand]:
        """Evaluates all 56 combinations in an 8-card hand and ranks by raw score."""
        if not full_hand:
            return [EvaluatedHand("High Card", [], 5, 1, 0, 5)]
        n = len(full_hand)
        size = min(5, n)
        all_combos = itertools.combinations(full_hand, size)
        evaluated = [cls.evaluate_5card_score(list(combo)) for combo in all_combos]
        evaluated.sort(key=lambda x: x.total_score, reverse=True)
        return evaluated[:top_k]

    @classmethod
    def recommend_discards(cls, full_hand: List[Card]) -> List[Card]:
        """Identifies optimal cards to discard (e.g. digging for flush or weak unaligned ranks)."""
        suit_counts = Counter(c.suit for c in full_hand)
        most_common_suit, count = suit_counts.most_common(1)[0]

        # If holding 3 or 4 cards of one suit, discard the non-matching cards
        if 3 <= count < 5:
            discards = [c for c in full_hand if c.suit != most_common_suit]
            # Keep highest value non-flush cards if limit is 5
            return discards[:5]

        # Fallback: Discard lowest rank cards that are not part of pairs/trips
        rank_counts = Counter(c.value for c in full_hand)
        unpaired = [c for c in full_hand if rank_counts[c.value] == 1]
        unpaired.sort(key=lambda c: c.value)
        return unpaired[:3]
