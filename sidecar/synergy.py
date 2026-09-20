"""
Joker Synergy Analysis Engine for Balatro Agent.

Parses joker effect text to extract what each joker rewards (suits, hand types,
card counts, resources), then provides annotation methods to surface synergy
context in decision criteria.

Python analyzes synergies; Laya uses them to decide. No automatic decisions.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Patterns to extract synergy-relevant information from joker effect text
_SUIT_PATTERN = re.compile(
    r"\b(hearts?|diamonds?|clubs?|spades?)\b", re.IGNORECASE
)
_HAND_TYPE_PATTERN = re.compile(
    r"\b(flush|straight flush|straight|full house|two pair|"
    r"three of a kind|four of a kind|pair|high card)\b",
    re.IGNORECASE,
)
_CARD_COUNT_PATTERN = re.compile(
    r"(\d+)\s+or\s+fewer\s+cards?"
    r"|exactly\s+(\d+)\s+cards?"
    r"|contains?\s+(\d+)\s+cards?",
    re.IGNORECASE,
)
_FACE_CARD_PATTERN = re.compile(
    r"\bface\s+cards?\b|\bkings?\b|\bqueens?\b|\bjacks?\b", re.IGNORECASE
)
_RESOURCE_PATTERN = re.compile(
    r"\b(?:remaining|each)\s+(discard|hand)\b"
    r"|\bper\s+(discard|hand)\b"
    r"|\b(discard|hand)\w*\s+remaining\b",
    re.IGNORECASE,
)

# Canonical suit name normalization
_SUIT_CANONICAL = {
    "heart": "Hearts", "hearts": "Hearts",
    "diamond": "Diamonds", "diamonds": "Diamonds",
    "club": "Clubs", "clubs": "Clubs",
    "spade": "Spades", "spades": "Spades",
}

# Canonical hand type normalization
_HAND_TYPE_CANONICAL = {
    "flush": "Flush",
    "straight flush": "Straight Flush",
    "straight": "Straight",
    "full house": "Full House",
    "two pair": "Two Pair",
    "three of a kind": "Three of a Kind",
    "four of a kind": "Four of a Kind",
    "pair": "Pair",
    "high card": "High Card",
}


@dataclass
class JokerSynergy:
    """What a single joker rewards."""
    name: str
    effect: str
    suits: List[str] = field(default_factory=list)
    hand_types: List[str] = field(default_factory=list)
    card_counts: List[int] = field(default_factory=list)
    face_cards: bool = False
    resources: List[str] = field(default_factory=list)

    @property
    def tags(self) -> List[str]:
        """Human-readable list of what this joker rewards."""
        parts = []
        for s in self.suits:
            parts.append(s)
        for h in self.hand_types:
            parts.append(h)
        if self.face_cards:
            parts.append("Face cards")
        for c in self.card_counts:
            parts.append(f"≤{c} cards")
        for r in self.resources:
            parts.append(f"{r}s remaining")
        return parts


@dataclass
class SynergyProfile:
    """Aggregate synergy profile from all current jokers."""
    favored_suits: List[str] = field(default_factory=list)
    favored_hand_types: List[str] = field(default_factory=list)
    favored_card_counts: List[int] = field(default_factory=list)
    favored_face_cards: bool = False
    resource_bonuses: List[str] = field(default_factory=list)
    joker_synergies: Dict[str, JokerSynergy] = field(default_factory=dict)

    @property
    def has_synergies(self) -> bool:
        return bool(
            self.favored_suits
            or self.favored_hand_types
            or self.favored_card_counts
            or self.favored_face_cards
            or self.resource_bonuses
        )

    @property
    def summary(self) -> str:
        """Short summary of what the joker lineup rewards."""
        parts = []
        if self.favored_suits:
            parts.append(", ".join(self.favored_suits))
        if self.favored_hand_types:
            parts.append(", ".join(self.favored_hand_types))
        if self.favored_face_cards:
            parts.append("Face cards")
        if self.favored_card_counts:
            parts.append(", ".join(f"≤{c} cards" for c in self.favored_card_counts))
        return ", ".join(parts) if parts else "no specific synergy"


class JokerSynergyAnalyzer:
    """Analyzes joker effects and surfaces synergy context for decision criteria."""

    @staticmethod
    def _extract_effect(joker: Dict[str, Any]) -> str:
        """Extract effect text from a joker dict, handling both data formats."""
        val_obj = joker.get("value", {}) if isinstance(joker, dict) else {}
        if isinstance(val_obj, dict):
            return (val_obj.get("effect") or joker.get("effect") or "").strip()
        return (joker.get("effect") or "").strip()

    @staticmethod
    def _extract_name(joker: Dict[str, Any]) -> str:
        """Extract joker name from a joker dict."""
        return (joker.get("label") or joker.get("key") or "Joker").strip()

    def analyze_joker(self, joker: Dict[str, Any]) -> JokerSynergy:
        """Parse a single joker's effect text into structured synergy data."""
        name = self._extract_name(joker)
        effect = self._extract_effect(joker)
        effect_lower = effect.lower()

        suits = []
        for m in _SUIT_PATTERN.finditer(effect_lower):
            canonical = _SUIT_CANONICAL.get(m.group(1).lower())
            if canonical and canonical not in suits:
                suits.append(canonical)

        hand_types = []
        for m in _HAND_TYPE_PATTERN.finditer(effect_lower):
            canonical = _HAND_TYPE_CANONICAL.get(m.group(1).lower())
            if canonical and canonical not in hand_types:
                hand_types.append(canonical)

        card_counts = []
        for m in _CARD_COUNT_PATTERN.finditer(effect_lower):
            val = m.group(1) or m.group(2) or m.group(3)
            if val:
                card_counts.append(int(val))

        face_cards = bool(_FACE_CARD_PATTERN.search(effect_lower))

        resources = []
        for m in _RESOURCE_PATTERN.finditer(effect_lower):
            res = m.group(1) or m.group(2) or m.group(3)
            if res and res.lower() not in resources:
                resources.append(res.lower())

        return JokerSynergy(
            name=name,
            effect=effect,
            suits=suits,
            hand_types=hand_types,
            card_counts=card_counts,
            face_cards=face_cards,
            resources=resources,
        )

    def analyze(self, jokers: List[Dict[str, Any]]) -> SynergyProfile:
        """Build aggregate synergy profile from all current jokers."""
        profile = SynergyProfile()

        for joker in jokers:
            if not isinstance(joker, dict):
                continue
            # Skip debuffed jokers
            state = joker.get("state", {})
            if isinstance(state, dict) and state.get("debuff"):
                continue

            synergy = self.analyze_joker(joker)
            if synergy.tags:  # Only track jokers with detectable synergies
                profile.joker_synergies[synergy.name] = synergy

            for s in synergy.suits:
                if s not in profile.favored_suits:
                    profile.favored_suits.append(s)
            for h in synergy.hand_types:
                if h not in profile.favored_hand_types:
                    profile.favored_hand_types.append(h)
            for c in synergy.card_counts:
                if c not in profile.favored_card_counts:
                    profile.favored_card_counts.append(c)
            if synergy.face_cards:
                profile.favored_face_cards = True
            for r in synergy.resources:
                if r not in profile.resource_bonuses:
                    profile.resource_bonuses.append(r)

        return profile

    def annotate_hand(
        self,
        profile: SynergyProfile,
        cards: List[Any],
        hand_type: str,
    ) -> str:
        """Return a synergy annotation for a hand candidate, or '' if no match.

        Args:
            profile: Current synergy profile from analyze()
            cards: List of Card objects (must have .suit attribute)
            hand_type: Identified hand type string (e.g. "Flush", "Pair")
        """
        if not profile.has_synergies:
            return ""

        matches = []

        # Check hand type synergies
        if hand_type in profile.favored_hand_types:
            matching_jokers = [
                name for name, syn in profile.joker_synergies.items()
                if hand_type in syn.hand_types
            ]
            if matching_jokers:
                matches.append(f"{hand_type} → {', '.join(matching_jokers)}")

        # Check suit synergies
        card_suits = [getattr(c, "suit", "") for c in cards]
        for suit in profile.favored_suits:
            count = sum(1 for s in card_suits if s == suit)
            if count > 0:
                matching_jokers = [
                    name for name, syn in profile.joker_synergies.items()
                    if suit in syn.suits
                ]
                if matching_jokers:
                    matches.append(f"{count}× {suit} → {', '.join(matching_jokers)}")

        # Check face card synergies
        if profile.favored_face_cards:
            face_ranks = {"J", "Q", "K"}
            face_count = sum(1 for c in cards if getattr(c, "rank", "") in face_ranks)
            if face_count > 0:
                matching_jokers = [
                    name for name, syn in profile.joker_synergies.items()
                    if syn.face_cards
                ]
                if matching_jokers:
                    matches.append(f"{face_count}× Face → {', '.join(matching_jokers)}")

        # Check card count synergies
        n_cards = len(cards)
        for limit in profile.favored_card_counts:
            if n_cards <= limit:
                matching_jokers = [
                    name for name, syn in profile.joker_synergies.items()
                    if limit in syn.card_counts
                ]
                if matching_jokers:
                    matches.append(f"≤{limit} cards → {', '.join(matching_jokers)}")

        if not matches:
            return ""
        return "🃏 " + "; ".join(matches)

    def annotate_shop_item(
        self,
        profile: SynergyProfile,
        item_effect: str,
        item_name: str,
    ) -> str:
        """Return a synergy annotation for a shop item, or '' if no synergies detected.

        Checks whether the item's effect OVERLAPS with existing joker synergies
        (reinforces current build) or introduces NEW synergies.
        """
        if not item_effect:
            return ""

        item_synergy = self.analyze_joker({"label": item_name, "value": {"effect": item_effect}})
        if not item_synergy.tags:
            return ""

        # Check overlap with existing profile
        overlaps = []
        new_synergies = []

        for s in item_synergy.suits:
            if s in profile.favored_suits:
                overlaps.append(s)
            else:
                new_synergies.append(s)
        for h in item_synergy.hand_types:
            if h in profile.favored_hand_types:
                overlaps.append(h)
            else:
                new_synergies.append(h)
        if item_synergy.face_cards:
            if profile.favored_face_cards:
                overlaps.append("Face cards")
            else:
                new_synergies.append("Face cards")

        parts = []
        if overlaps:
            parts.append(f"reinforces your {', '.join(overlaps)} build")
        if new_synergies:
            parts.append(f"adds {', '.join(new_synergies)} synergy")
        if item_synergy.tags and not overlaps and not new_synergies:
            parts.append(f"rewards {', '.join(item_synergy.tags)}")

        if not parts:
            return ""
        return " — 🃏 " + "; ".join(parts)

    def joker_contribution(
        self,
        profile: SynergyProfile,
        joker_name: str,
        joker_effect: str,
    ) -> str:
        """Describe what synergies would be lost by selling this joker."""
        synergy = profile.joker_synergies.get(joker_name)
        if not synergy or not synergy.tags:
            return "no detected synergies"
        return ", ".join(synergy.tags) + " synergy"
