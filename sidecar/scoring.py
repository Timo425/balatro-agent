"""Score estimation and plain-language labels for Balatro decisions.

Everything here is a FACT about the current board: what a candidate hand would
score, how that compares with the blind's target, what the jokers add. Facts are
handed to the decision model so it never has to do arithmetic -- but the model,
not this module, picks the move. See CLAUDE.md, "Python never decides".
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sidecar.poker_eval import BASE_HAND_SCORES, Card, PokerEvaluator

# Numeric joker effects we can model. Anything else is reported as "not modelled"
# rather than silently ignored, so the uncertainty reaches the model as a fact.
_XMULT_RE = re.compile(r"(?<![A-Za-z])[xX]\s*(\d+(?:\.\d+)?)")
_FLAT_MULT_RE = re.compile(r"\+\s*(\d+(?:\.\d+)?)\s*mult", re.IGNORECASE)
_CHIPS_RE = re.compile(r"\+\s*(\d+)\s*chips?", re.IGNORECASE)
# "each/per/every", and Balatro's other per-card phrasing: "Played <x> cards give ...".
_PER_RE = re.compile(r"\b(each|per|every)\b|\bcards?\b[^.]*\bgives?\b", re.IGNORECASE)
_RANDOM_RE = re.compile(r"\b\d+\s+in\s+\d+\b|\bchance\b|\brandom\b", re.IGNORECASE)

FACE_RANKS = {"J", "Q", "K"}


@dataclass
class ScoreEstimate:
    """Estimated score of one candidate play, with its joker breakdown."""

    hand_type: str
    chips: int
    mult: float
    total: int
    base_total: int
    joker_notes: List[str] = field(default_factory=list)
    unmodelled: List[str] = field(default_factory=list)

    @property
    def has_jokers(self) -> bool:
        return bool(self.joker_notes)

    @property
    def uncertain(self) -> bool:
        return bool(self.unmodelled)


def hand_level_values(hands_data: Optional[Dict[str, Any]], hand_type: str) -> tuple:
    """Live (chips, mult) for a hand type, falling back to vanilla level 1."""
    default_c, default_m = BASE_HAND_SCORES.get(hand_type, (10, 2))
    info = (hands_data or {}).get(hand_type)
    if not isinstance(info, dict):
        return default_c, default_m, 1
    return (
        info.get("chips", default_c),
        info.get("mult", default_m),
        info.get("level", 1),
    )


def _is_random_gated(effect: str) -> bool:
    """True when a random chance gates the whole effect.

    Cavendish reads "X3 Mult, 1 in 1000 chance to go extinct": the multiplier is
    certain and only the downside is random, so position decides. A text whose
    chance clause comes before every number ("1 in 4 chance for +10 Mult") is gated.
    """
    m_rand = _RANDOM_RE.search(effect)
    if not m_rand:
        return False
    first_number = re.search(r"\d", effect)
    return first_number is None or m_rand.start() <= first_number.start()


def _joker_effect(joker: Dict[str, Any]) -> str:
    val = joker.get("value") if isinstance(joker, dict) else None
    if isinstance(val, dict) and val.get("effect"):
        return str(val["effect"])
    return str(joker.get("effect") or "") if isinstance(joker, dict) else ""


def _joker_name(joker: Dict[str, Any]) -> str:
    if not isinstance(joker, dict):
        return "Joker"
    return str(joker.get("label") or joker.get("key") or "Joker")


def _is_debuffed(joker: Dict[str, Any]) -> bool:
    state = joker.get("state") if isinstance(joker, dict) else None
    return bool(isinstance(state, dict) and state.get("debuff"))


def _condition_multiplier(
    effect: str,
    hand_type: str,
    played_cards: List[Card],
    scoring_cards: List[Card],
    hands_left: int,
    discards_left: int,
) -> Optional[float]:
    """How many times a joker's numeric effect applies, or None when it does not.

    Returns a count (0 means the condition is present but unmet).
    """
    from sidecar.synergy import JokerSynergyAnalyzer

    syn = JokerSynergyAnalyzer().analyze_joker({"label": "", "value": {"effect": effect}})
    per_card = bool(_PER_RE.search(effect))

    if syn.resources:
        left = discards_left if "discard" in syn.resources else hands_left
        return float(max(0, left))

    if syn.hand_types:
        return 1.0 if hand_type in syn.hand_types else 0.0

    if syn.suits:
        matched = [c for c in scoring_cards if c.suit in syn.suits]
        if per_card:
            return float(len(matched))
        return 1.0 if matched else 0.0

    if syn.face_cards:
        faces = [c for c in scoring_cards if c.rank in FACE_RANKS]
        if per_card:
            return float(len(faces))
        return 1.0 if faces else 0.0

    if syn.card_counts:
        limit = min(syn.card_counts)
        return 1.0 if len(played_cards) <= limit else 0.0

    # No condition detected: the effect always applies.
    return 1.0


def estimate_play(
    played_cards: List[Card],
    hands_data: Optional[Dict[str, Any]] = None,
    jokers: Optional[List[Dict[str, Any]]] = None,
    hands_left: int = 0,
    discards_left: int = 0,
) -> ScoreEstimate:
    """Estimated chips scored by playing `played_cards`, jokers included.

    Joker effects that cannot be read off their text (retriggers, scaling counters,
    random chances) are listed in `unmodelled` so the caller can tell the model that
    the number is a floor rather than pretending certainty.
    """
    hand_type, scoring_cards = PokerEvaluator.identify_hand(played_cards)
    base_c, base_m, _level = hand_level_values(hands_data, hand_type)
    chips = base_c + PokerEvaluator.scoring_chips(scoring_cards)
    mult = float(base_m)
    base_total = int(chips * mult)

    notes: List[str] = []
    unmodelled: List[str] = []

    for joker in jokers or []:
        if not isinstance(joker, dict) or _is_debuffed(joker):
            continue
        name = _joker_name(joker)
        effect = _joker_effect(joker)
        if not effect:
            unmodelled.append(name)
            continue
        gated = _is_random_gated(effect)
        if gated:
            unmodelled.append(name)
            continue

        times = _condition_multiplier(
            effect, hand_type, played_cards, scoring_cards, hands_left, discards_left
        )
        if times is None or times <= 0:
            continue

        applied = False
        m_chips = _CHIPS_RE.search(effect)
        if m_chips:
            gain = int(int(m_chips.group(1)) * times)
            if gain:
                chips += gain
                notes.append(f"{name} +{gain} chips")
                applied = True

        m_mult = _FLAT_MULT_RE.search(effect)
        if m_mult:
            gain = float(m_mult.group(1)) * times
            if gain:
                mult += gain
                notes.append(f"{name} +{gain:g} mult")
                applied = True

        # `X3 Mult` must not be confused with the `x` inside words, so only accept it
        # when the text has no `+N Mult` reading and the multiplier is at least 1.
        if not m_mult:
            m_x = _XMULT_RE.search(effect)
            if m_x and float(m_x.group(1)) >= 1.0 and "mult" in effect.lower():
                factor = float(m_x.group(1))
                mult *= factor
                notes.append(f"{name} x{factor:g} mult")
                applied = True

        if not applied:
            unmodelled.append(name)

    return ScoreEstimate(
        hand_type=hand_type,
        chips=int(chips),
        mult=mult,
        total=int(chips * mult),
        base_total=base_total,
        joker_notes=notes,
        unmodelled=unmodelled,
    )


def strength_label(score: int, needed: int) -> str:
    """Absolute quality of one option against the blind's remaining target.

    Deliberately NOT comparative: it says how this option stands on its own, never
    that it is the best or second-best available. Ranking the options is the
    decision model's job.
    """
    if needed <= 0:
        return "WINS"
    if score >= needed:
        return "WINS"
    pct = score * 100 // needed
    if pct >= 60:
        return "STRONG"
    if pct >= 30:
        return "OK"
    if pct >= 10:
        return "WEAK"
    return "TINY"


def pace_label(score: int, needed: int, hands_left: int) -> str:
    """Whether repeating a play of this size would finish the blind.

    A per-option fact (`this hand, repeated`), so it never compares two options.
    """
    if needed <= 0 or score >= needed:
        return "wins now"
    if hands_left <= 1:
        return "last hand, falls short"
    if score * hands_left >= needed:
        return f"needs {min(hands_left, -(-needed // max(1, score)))} hands like it"
    return f"{hands_left} more like it still lose"


def describe_estimate(est: ScoreEstimate, needed: int, hands_left: int) -> str:
    """Short model-facing summary of one candidate: `~108 chips, WEAK, 3 more like it still lose`."""
    parts = [f"~{est.total} chips", strength_label(est.total, needed), pace_label(est.total, needed, hands_left)]
    if est.uncertain:
        parts.append("plus unread joker text")
    return ", ".join(parts)
