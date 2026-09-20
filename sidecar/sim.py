"""Offline Balatro round simulator -- an EVALUATION HARNESS, not a player.

It deals cards, applies the game's scoring rules to whatever move it is given,
and reports whether the blind was beaten. It asks `BalatroAgent` for every move
exactly as the live runner does, so a decision tree can be exercised thousands of
times without launching the game.

It contains no strategy: `step()` executes the action it is handed. The policies
that answer the questions live in `sidecar/stress.py` and are measuring
instruments -- they must never be wired into the live runner. See CLAUDE.md.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sidecar.agent import AgentAction, BalatroAgent, BalatroGameState
from sidecar.poker_eval import BASE_HAND_SCORES, RANK_VALUES, Card, PokerEvaluator

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# White-stake base chips per ante; small = base, big = 1.5x, boss = 2x.
ANTE_BASE_CHIPS = {1: 300, 2: 800, 3: 2000, 4: 5000, 5: 11000, 6: 20000, 7: 35000, 8: 50000}
BLIND_MULTIPLIER = {"Small Blind": 1.0, "Big Blind": 1.5, "Boss Blind": 2.0}
BLIND_ORDER = ["Small Blind", "Big Blind", "Boss Blind"]

FACE_RANKS = {"J", "Q", "K"}


def blind_target(ante: int, blind: str) -> int:
    base = ANTE_BASE_CHIPS.get(ante, ANTE_BASE_CHIPS[8] * (2 ** (ante - 8)))
    return int(base * BLIND_MULTIPLIER[blind])


@dataclass
class SimJoker:
    """A joker with both the text the model reads and the rule the simulator applies.

    Keeping the two separate is deliberate: it means the simulator's ground truth
    never comes from the same text parsing that `sidecar/scoring.py` shows the
    model, so a stress run also measures how far the model-facing estimate drifts
    from what the game would actually pay.
    """

    name: str
    effect: str
    apply: Callable[[str, List[Card], List[Card], int, int], Tuple[int, float]]

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.name, "key": self.name.lower().replace(" ", "_"),
                "value": {"effect": self.effect}, "cost": {"sell": 2}}


def _suit_mult(suit: str, amount: int) -> Callable[..., Tuple[int, float]]:
    def rule(hand_type, played, scoring, hands_left, discards_left):
        return 0, amount * sum(1 for c in scoring if c.suit == suit)
    return rule


def _hand_type_mult(target_type: str, amount: int) -> Callable[..., Tuple[int, float]]:
    def rule(hand_type, played, scoring, hands_left, discards_left):
        return 0, (amount if hand_type == target_type else 0)
    return rule


JOKER_LIBRARY: Dict[str, SimJoker] = {
    "joker": SimJoker("Joker", "+4 Mult",
                      lambda ht, p, s, hl, dl: (0, 4.0)),
    "greedy": SimJoker("Greedy Joker", "Played cards with Diamond suit give +3 Mult when scored",
                       _suit_mult("Diamonds", 3)),
    "lusty": SimJoker("Lusty Joker", "Played cards with Heart suit give +3 Mult when scored",
                      _suit_mult("Hearts", 3)),
    "half": SimJoker("Half Joker", "+20 Mult if played hand contains 3 or fewer cards",
                     lambda ht, p, s, hl, dl: (0, 20.0 if len(p) <= 3 else 0.0)),
    "banner": SimJoker("Banner", "+30 Chips for each remaining discard",
                       lambda ht, p, s, hl, dl: (30 * max(0, dl), 0.0)),
    "scary": SimJoker("Scary Face", "Played face cards give +30 Chips",
                      lambda ht, p, s, hl, dl: (30 * sum(1 for c in s if c.rank in FACE_RANKS), 0.0)),
    "sly": SimJoker("Sly Joker", "+50 Chips if played hand contains a Pair",
                    lambda ht, p, s, hl, dl: (50 if ht in ("Pair", "Two Pair", "Full House",
                                                           "Three of a Kind", "Four of a Kind") else 0, 0.0)),
    "cavendish": SimJoker("Cavendish", "X3 Mult, 1 in 1000 chance to go extinct",
                          lambda ht, p, s, hl, dl: (0, 0.0)),  # x-mult applied separately below
}


@dataclass
class RoundResult:
    ante: int
    blind: str
    target: int
    chips: int
    won: bool
    hands_used: int
    discards_used: int
    actions: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    seed: int
    rounds: List[RoundResult] = field(default_factory=list)
    fallbacks: int = 0

    @property
    def rounds_won(self) -> int:
        return sum(1 for r in self.rounds if r.won)

    @property
    def furthest_ante(self) -> int:
        return max((r.ante for r in self.rounds if r.won), default=0)

    @property
    def last_round(self) -> Optional[RoundResult]:
        return self.rounds[-1] if self.rounds else None


class BalatroSim:
    """Deals, scores and advances blinds. Executes moves; never chooses them."""

    def __init__(
        self,
        seed: int = 0,
        hand_size: int = 8,
        hands_per_round: int = 4,
        discards_per_round: int = 4,
        jokers: Optional[Sequence[SimJoker]] = None,
        hand_levels: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.rng = random.Random(seed)
        self.seed = seed
        self.hand_size = hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.jokers = list(jokers or [])
        self.hand_levels = dict(hand_levels or {})

    # -- deck -----------------------------------------------------------
    def fresh_deck(self) -> List[Card]:
        deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.rng.shuffle(deck)
        return deck

    def _levels_payload(self) -> Dict[str, Dict[str, int]]:
        payload = {}
        for name, (chips, mult) in BASE_HAND_SCORES.items():
            lvl_chips, lvl_mult = self.hand_levels.get(name, (chips, mult))
            payload[name] = {"chips": lvl_chips, "mult": lvl_mult, "level": 1}
        return payload

    # -- scoring (ground truth) -----------------------------------------
    def score(self, played: List[Card], hands_left: int, discards_left: int) -> int:
        hand_type, scoring = PokerEvaluator.identify_hand(played)
        base_chips, base_mult = self.hand_levels.get(hand_type, BASE_HAND_SCORES[hand_type])
        chips = base_chips + PokerEvaluator.scoring_chips(scoring)
        mult = float(base_mult)
        for joker in self.jokers:
            add_chips, add_mult = joker.apply(hand_type, played, scoring, hands_left, discards_left)
            chips += add_chips
            mult += add_mult
            if joker.name == "Cavendish":
                mult *= 3
        return int(chips * mult)

    # -- round ----------------------------------------------------------
    def play_round(self, agent: BalatroAgent, ante: int, blind: str) -> RoundResult:
        target = blind_target(ante, blind)
        deck = self.fresh_deck()
        hand: List[Card] = []
        chips = 0
        hands_left = self.hands_per_round
        discards_left = self.discards_per_round
        result = RoundResult(ante=ante, blind=blind, target=target, chips=0, won=False,
                             hands_used=0, discards_used=0)

        def draw() -> None:
            while len(hand) < self.hand_size and deck:
                hand.append(deck.pop())
            for i, card in enumerate(hand):
                card.id = i

        draw()
        while hands_left > 0 and chips < target and hand:
            state = BalatroGameState(
                ante=ante, blind_name=blind, target_chips=target, current_chips=chips,
                hands_left=hands_left, discards_left=discards_left, money=4,
                hand=list(hand), jokers=[j.name for j in self.jokers],
            )
            raw_g = {
                "state": "SELECTING_HAND",
                "ante_num": ante,
                "hands": self._levels_payload(),
                "jokers": {"cards": [j.as_dict() for j in self.jokers],
                           "count": len(self.jokers), "max": 5},
            }
            action: AgentAction = agent.decide_hand_action(state, raw_g=raw_g)
            chosen_ids = {c.id for c in action.cards}
            chosen = [c for c in hand if c.id in chosen_ids]
            if not chosen:
                break

            if action.action_type == "DISCARD" and discards_left > 0:
                discards_left -= 1
                result.discards_used += 1
                result.actions.append(f"DISCARD {' '.join(c.short for c in chosen)}")
            else:
                gained = self.score(chosen[:5], hands_left, discards_left)
                chips += gained
                hands_left -= 1
                result.hands_used += 1
                result.actions.append(
                    f"PLAY {' '.join(c.short for c in chosen)} -> {gained}"
                )

            hand = [c for c in hand if c.id not in chosen_ids]
            draw()

        result.chips = chips
        result.won = chips >= target
        return result

    def play_run(self, agent: BalatroAgent, max_ante: int = 8) -> RunResult:
        """Plays blinds in order until one is lost. No shop: this measures the
        hand-level decision tree in isolation."""
        run = RunResult(seed=self.seed)
        for ante in range(1, max_ante + 1):
            for blind in BLIND_ORDER:
                round_result = self.play_round(agent, ante, blind)
                run.rounds.append(round_result)
                if not round_result.won:
                    run.fallbacks = getattr(agent.backend, "fallbacks", 0)
                    return run
        run.fallbacks = getattr(agent.backend, "fallbacks", 0)
        return run
