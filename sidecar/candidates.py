"""Enumeration of the legal moves a Balatro hand offers, annotated with facts.

The rule this module lives by: it ENUMERATES and DESCRIBES, it never ranks,
filters by quality, or picks. Options come back in the game's own fixed order
(poker-hand ranking, then rank order) so that position carries no hint about
which option Python thinks is better. See CLAUDE.md, "Python never decides".
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sidecar.poker_eval import HAND_TYPE_ORDER, Card, PokerEvaluator
from sidecar.scoring import ScoreEstimate, describe_estimate, estimate_play, strength_label

MAX_DISCARD = 5
MAX_PLAY = 5

_RANK_WORDS = {
    "A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks", "10": "10s",
    "9": "9s", "8": "8s", "7": "7s", "6": "6s", "5": "5s", "4": "4s",
    "3": "3s", "2": "2s",
}


def _plural(rank: str) -> str:
    return _RANK_WORDS.get(rank, f"{rank}s")


@dataclass
class PlayOption:
    """One legal hand the player could play right now."""

    key: str
    hand_type: str
    core: List[Card]                 # the cards that actually score
    filler: List[Card] = field(default_factory=list)   # extra cards played but not scored
    estimate: Optional[ScoreEstimate] = None
    note: str = ""

    @property
    def cards(self) -> List[Card]:
        return self.core + self.filler

    def label(self, needed: int, hands_left: int) -> str:
        """Short, non-comparative model-facing text, verdict first.

        `WINS - Flush AS KS QS JS 9S: ~340 chips, wins now`. The verdict leads
        because option text is clipped from the end under a tight head budget,
        and a word is a far stronger signal to a pattern matcher than a number.
        It describes this option against the blind, never against the others.
        """
        cards = " ".join(c.short for c in self.core)
        body = f"{self.hand_type} {cards}"
        if self.estimate is None:
            return body
        verdict = strength_label(self.estimate.total, needed)
        return f"{verdict} - {body}: {describe_estimate(self.estimate, needed, hands_left)}"


@dataclass
class DiscardPlan:
    """One named way of throwing cards away, named by the hand it is digging for.

    `target_hand` is what the plan is trying to build. Naming it turns the
    play-or-dig call into a comparison between poker hands -- which is the one
    thing a pattern matcher with a language prior is reliably good at -- instead
    of an abstract yes/no about discarding.
    """

    key: str
    discard: List[Card]
    keep_reason: str
    target_hand: str = ""

    def label(self, discards_left: int) -> str:
        thrown = " ".join(c.short for c in self.discard)
        return f"{self.keep_reason}; throw {len(self.discard)} ({thrown})"

    def aim_label(self) -> str:
        """How this plan reads next to a hand that can be played right now."""
        aim = self.target_hand or "a better hand"
        return f"Dig for {aim}: {self.keep_reason}, throw {len(self.discard)}"


def _straight_runs(hand: Sequence[Card]) -> List[List[Card]]:
    """Every distinct 5-card straight available, including the A-2-3-4-5 wheel."""
    by_value: Dict[int, Card] = {}
    for c in sorted(hand, key=lambda c: c.chip_value, reverse=True):
        by_value.setdefault(c.value, c)
    values = sorted(by_value)
    runs: List[List[Card]] = []
    for start in values:
        window = [start + i for i in range(5)]
        if all(v in by_value for v in window):
            runs.append([by_value[v] for v in reversed(window)])
    wheel = [14, 5, 4, 3, 2]
    if all(v in by_value for v in wheel):
        runs.append([by_value[v] for v in wheel])
    return runs


def _suit_groups(hand: Sequence[Card], minimum: int) -> List[Tuple[str, List[Card]]]:
    groups = []
    for suit in sorted({c.suit for c in hand}):
        cards = sorted([c for c in hand if c.suit == suit], key=lambda c: c.chip_value, reverse=True)
        if len(cards) >= minimum:
            groups.append((suit, cards))
    return groups


def enumerate_play_cores(hand: Sequence[Card]) -> List[PlayOption]:
    """Every distinct scoring core the hand can make, in the game's hand-rank order.

    "Distinct" means a different set of *scoring* cards: two pairs of 9s vs 8s are
    separate options, but the kickers alongside them are not -- kickers score
    nothing, so they are offered separately as the `filler` decision.

    Where one candidate is strictly dominated by another of the same hand type
    (the same five-card flush but with a lower card), only the higher-chip version
    is enumerated. That is card-game dominance, not a strategic preference.
    """
    hand = list(hand)
    if not hand:
        return []

    options: List[PlayOption] = []
    rank_counts = Counter(c.value for c in hand)
    by_value: Dict[int, List[Card]] = {}
    for c in hand:
        by_value.setdefault(c.value, []).append(c)
    for v in by_value:
        by_value[v].sort(key=lambda c: c.chip_value, reverse=True)

    def rank_of(value: int) -> str:
        return by_value[value][0].rank

    def add(key: str, cards: List[Card]) -> None:
        hand_type, scoring = PokerEvaluator.identify_hand(cards)
        options.append(PlayOption(key=key, hand_type=hand_type, core=list(cards)))

    straights = _straight_runs(hand)

    # Straight Flush
    for run in straights:
        if len({c.suit for c in run}) == 1:
            add(f"straight_flush_{run[0].suit.lower()}", run)

    # Four of a Kind
    for value in sorted((v for v, n in rank_counts.items() if n >= 4), reverse=True):
        add(f"four_{rank_of(value).lower()}", by_value[value][:4])

    # Full House
    trips = sorted((v for v, n in rank_counts.items() if n >= 3), reverse=True)
    pairs = sorted((v for v, n in rank_counts.items() if n >= 2), reverse=True)
    for t in trips:
        for p in pairs:
            if p == t:
                continue
            add(f"full_house_{rank_of(t).lower()}_{rank_of(p).lower()}",
                by_value[t][:3] + by_value[p][:2])
            break  # one full house per trip rank is enough; the pair rank adds no chips choice

    # Flush (five highest of each suit that has five)
    for suit, cards in _suit_groups(hand, 5):
        add(f"flush_{suit.lower()}", cards[:5])

    # Straight
    for run in straights:
        if len({c.suit for c in run}) > 1:
            add(f"straight_{run[0].rank.lower()}", run)

    # Three of a Kind
    for value in sorted((v for v, n in rank_counts.items() if n >= 3), reverse=True):
        add(f"three_{rank_of(value).lower()}", by_value[value][:3])

    # Two Pair
    pair_values = sorted((v for v, n in rank_counts.items() if n >= 2), reverse=True)
    for i in range(len(pair_values)):
        for j in range(i + 1, len(pair_values)):
            hi, lo = pair_values[i], pair_values[j]
            add(f"two_pair_{rank_of(hi).lower()}_{rank_of(lo).lower()}",
                by_value[hi][:2] + by_value[lo][:2])

    # Pair
    for value in pair_values:
        add(f"pair_{rank_of(value).lower()}", by_value[value][:2])

    # High Card: the highest card, and (when they differ) the lowest, which is the
    # cheapest legal way to spend a hand and cycle a dead card.
    ordered = sorted(hand, key=lambda c: (c.chip_value, c.value))
    add("high_card", [ordered[-1]])
    if len(ordered) > 1 and ordered[0].id != ordered[-1].id:
        add("dump_low_card", [ordered[0]])

    # Stable dedupe on the scoring core.
    seen = set()
    unique: List[PlayOption] = []
    for opt in options:
        sig = (opt.hand_type, tuple(sorted(c.id for c in opt.core)))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(opt)

    order = {name: i for i, name in enumerate(HAND_TYPE_ORDER)}
    unique.sort(key=lambda o: (order.get(o.hand_type, 99), -sum(c.chip_value for c in o.core)))
    return unique


def annotate_play_options(
    options: Sequence[PlayOption],
    hands_data: Optional[Dict[str, Any]] = None,
    jokers: Optional[List[Dict[str, Any]]] = None,
    hands_left: int = 0,
    discards_left: int = 0,
) -> List[PlayOption]:
    """Attaches a score estimate to each option. Does not reorder or drop any."""
    for opt in options:
        opt.estimate = estimate_play(
            opt.cards, hands_data, jokers, hands_left=hands_left, discards_left=discards_left
        )
    return list(options)


def pick_filler(hand: Sequence[Card], core: Sequence[Card]) -> List[Card]:
    """The spare cards that would ride along with a short hand, lowest chips first.

    Mechanical tie-break, not a strategic choice: whether to send filler at all is
    the model's decision; if it says yes, the cheapest spare cards are the ones
    that cost the least to cycle away.
    """
    core_ids = {c.id for c in core}
    spare = sorted((c for c in hand if c.id not in core_ids), key=lambda c: (c.chip_value, c.value))
    return spare[: max(0, MAX_PLAY - len(core))]


def enumerate_discard_plans(
    hand: Sequence[Card],
    favored_suits: Optional[Sequence[str]] = None,
    favored_face_cards: bool = False,
) -> List[DiscardPlan]:
    """Named discard patterns whose precondition currently holds.

    Each plan is a recognisable shape ("keep the flush draw", "keep the pairs"),
    which is what a pattern-matching model can actually reason about -- far more
    than a list of card indices.
    """
    hand = list(hand)
    if len(hand) < 2:
        return []

    plans: List[DiscardPlan] = []
    suit_counts = Counter(c.suit for c in hand)
    rank_counts = Counter(c.value for c in hand)
    low_first = sorted(hand, key=lambda c: (c.chip_value, c.value))

    def add(key: str, keep: Sequence[Card], reason: str, target: str = "") -> None:
        keep_ids = {c.id for c in keep}
        throw = [c for c in low_first if c.id not in keep_ids][:MAX_DISCARD]
        if throw:
            plans.append(DiscardPlan(key=key, discard=throw, keep_reason=reason,
                                     target_hand=target))

    # Flush draw: three or four of one suit.
    for suit, count in sorted(suit_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if 3 <= count < 5:
            keep = [c for c in hand if c.suit == suit]
            away = 5 - count
            add(f"flush_draw_{suit.lower()}",
                keep,
                f"keep the {count} {suit}, {away} short",
                target=f"a Flush in {suit}")
            break

    # Straight draw: four cards inside a five-rank window.
    values = sorted({c.value for c in hand})
    best_window: List[int] = []
    for start in values:
        window = [v for v in values if start <= v <= start + 4]
        if len(window) > len(best_window):
            best_window = window
    if len(best_window) == 4:
        keep = []
        used = set()
        for v in best_window:
            for c in hand:
                if c.value == v and c.id not in used:
                    keep.append(c)
                    used.add(c.id)
                    break
        add("straight_draw", keep, "keep the 4 cards in a row, 1 short",
            target="a Straight")

    # Keep every pair/trip, throw the singles.
    paired = [c for c in hand if rank_counts[c.value] >= 2]
    if paired and len(paired) < len(hand):
        names = ", ".join(sorted({_plural(c.rank) for c in paired}))
        biggest = max(rank_counts[c.value] for c in paired)
        distinct = len({c.value for c in paired})
        if biggest >= 3:
            target = "Four of a Kind"
        elif distinct >= 2:
            target = "a Full House"
        else:
            target = "Three of a Kind"
        add("keep_pairs", paired, f"keep the {names}", target=target)

    # Keep the suits the jokers reward.
    for suit in favored_suits or []:
        if suit_counts.get(suit, 0) >= 2:
            keep = [c for c in hand if c.suit == suit]
            add(f"keep_joker_suit_{suit.lower()}", keep,
                f"keep the {suit} your jokers reward", target=f"more {suit} for your jokers")
            break

    # Keep the face cards the jokers reward.
    if favored_face_cards:
        faces = [c for c in hand if c.rank in ("J", "Q", "K")]
        if faces and len(faces) < len(hand):
            add("keep_faces", faces, "keep the face cards your jokers reward",
                target="more face cards for your jokers")

    # Always legal: throw the lowest cards.
    add("throw_lowest", low_first[MAX_DISCARD:], "keep the high cards",
        target="anything better")

    seen = set()
    unique: List[DiscardPlan] = []
    for plan in plans:
        sig = tuple(sorted(c.id for c in plan.discard))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(plan)
    return unique
