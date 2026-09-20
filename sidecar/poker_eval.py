"""Balatro poker mechanics: hand identification, *scoring-card* selection and chip math.

This module computes FACTS about the game's rules (what hand a set of cards makes,
which of those cards actually score, how many chips that is worth). It contains no
strategy and chooses no actions -- see CLAUDE.md, "Python never decides".
"""

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Card Rank mapping to numerical values
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "T": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14
}

# Chip values contributed by raw card ranks in Balatro
RANK_CHIPS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "T": 10,
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

# Canonical ordering of hand types, strongest first. This is the game's own fixed
# ranking of poker hands, NOT a situational judgement about which hand to play.
HAND_TYPE_ORDER = [
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
]

# How many cards of a played hand actually contribute their chips in Balatro.
# Non-scoring cards are played and discarded but add nothing to the score.
SCORING_CARD_COUNTS = {
    "Straight Flush": 5,
    "Four of a Kind": 4,
    "Full House": 5,
    "Flush": 5,
    "Straight": 5,
    "Three of a Kind": 3,
    "Two Pair": 4,
    "Pair": 2,
    "High Card": 1,
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

    @property
    def short(self) -> str:
        """Compact label such as `9S` / `10H` used in model-facing option text."""
        return f"{self.rank}{self.suit[0] if self.suit else '?'}"

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


@dataclass
class EvaluatedHand:
    hand_type: str
    cards: List[Card]
    scoring_cards: List[Card]
    base_chips: int
    base_mult: int
    card_chips: int
    total_score: int


def _straight_ranks(values: List[int]) -> Optional[List[int]]:
    """Returns the 5 rank values forming a straight, or None. Handles the A-2-3-4-5 wheel."""
    uniq = sorted(set(values), reverse=True)
    if len(uniq) < 5:
        return None
    for i in range(len(uniq) - 4):
        window = uniq[i:i + 5]
        if window[0] - window[4] == 4:
            return window
    if {14, 5, 4, 3, 2}.issubset(set(uniq)):
        return [5, 4, 3, 2, 14]
    return None


def _take_by_value(cards: List[Card], value: int, count: int) -> List[Card]:
    return [c for c in cards if c.value == value][:count]


class PokerEvaluator:
    """Hand identification and Balatro chip math. Facts only -- no strategy."""

    @staticmethod
    def identify_hand(cards: List[Card]) -> Tuple[str, List[Card]]:
        """Identifies the poker hand type and returns `(hand_name, scoring_cards)`.

        `scoring_cards` is the subset that actually earns chips in Balatro. This is
        the whole point of the function: a Pair played inside five cards scores only
        the two paired cards, so treating all five as scoring inflates the estimate
        by 2-3x and makes every downstream comparison wrong.
        """
        if not cards:
            return "High Card", []

        values = [c.value for c in cards]
        rank_counts = Counter(values)
        suit_counts = Counter(c.suit for c in cards)
        n = len(cards)

        by_count = sorted(rank_counts.items(), key=lambda kv: (-kv[1], -kv[0]))
        top_value, top_count = by_count[0]

        is_flush = n == 5 and len(suit_counts) == 1
        straight = _straight_ranks(values) if n == 5 else None

        if is_flush and straight:
            return "Straight Flush", list(cards)
        if top_count >= 4:
            return "Four of a Kind", _take_by_value(cards, top_value, 4)
        if top_count == 3 and len(by_count) > 1 and by_count[1][1] >= 2:
            return "Full House", list(cards)
        if is_flush:
            return "Flush", list(cards)
        if straight:
            return "Straight", list(cards)
        if top_count == 3:
            return "Three of a Kind", _take_by_value(cards, top_value, 3)

        pairs = [v for v, cnt in by_count if cnt == 2]
        if len(pairs) >= 2:
            hi, lo = pairs[0], pairs[1]
            return "Two Pair", _take_by_value(cards, hi, 2) + _take_by_value(cards, lo, 2)
        if len(pairs) == 1:
            return "Pair", _take_by_value(cards, pairs[0], 2)

        return "High Card", [max(cards, key=lambda c: c.value)]

    @staticmethod
    def scoring_chips(scoring_cards: List[Card]) -> int:
        """Chips contributed by the cards that actually score."""
        return sum(c.chip_value for c in scoring_cards)

    @classmethod
    def evaluate_5card_score(
        cls,
        cards: List[Card],
        level_chips: Optional[int] = None,
        level_mult: Optional[int] = None,
    ) -> EvaluatedHand:
        """Base (joker-free) score of a played set of cards.

        `level_chips` / `level_mult` accept the live in-game hand levels; when omitted
        the vanilla level-1 values are used.
        """
        hand_type, scoring_cards = cls.identify_hand(cards)
        default_chips, default_mult = BASE_HAND_SCORES[hand_type]
        base_chips = default_chips if level_chips is None else level_chips
        base_mult = default_mult if level_mult is None else level_mult
        card_chips = cls.scoring_chips(scoring_cards)
        total_score = (base_chips + card_chips) * base_mult
        return EvaluatedHand(
            hand_type=hand_type,
            cards=list(cards),
            scoring_cards=scoring_cards,
            base_chips=base_chips,
            base_mult=base_mult,
            card_chips=card_chips,
            total_score=total_score,
        )

    @classmethod
    def find_best_playable_hands(cls, full_hand: List[Card], top_k: int = 3) -> List[EvaluatedHand]:
        """Ranks every 5-card combination by raw score.

        ANALYSIS ONLY. This ranking must not be used to pick a move -- the decision
        model chooses from the enumerated candidates in `sidecar/candidates.py`.
        """
        if not full_hand:
            return [EvaluatedHand("High Card", [], [], 5, 1, 0, 5)]
        size = min(5, len(full_hand))
        evaluated = [cls.evaluate_5card_score(list(combo))
                     for combo in itertools.combinations(full_hand, size)]
        evaluated.sort(key=lambda x: x.total_score, reverse=True)
        return evaluated[:top_k]
