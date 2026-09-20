"""Compact, model-facing descriptions of the board.

Laya reads at most ~320 tokens of state on the English checkpoint (512 context
minus a 192-token option head) and ~768 on `typed-decisions`. The BalatroLLM
Jinja gamestate is many times that, and it is truncated from the end -- which
silently cut the player's own hand out of every prompt. These briefs put the
decision-relevant facts first and stay small enough to survive intact.

Facts only: no recommendation, no ordering by quality.
"""

from typing import Any, Dict, List, Optional, Sequence

from sidecar.candidates import DiscardPlan, PlayOption
from sidecar.poker_eval import Card


def _joker_line(jokers: Sequence[Dict[str, Any]]) -> str:
    if not jokers:
        return "Jokers: none"
    parts = []
    for j in jokers:
        if not isinstance(j, dict):
            continue
        name = j.get("label") or j.get("key") or "Joker"
        val = j.get("value") if isinstance(j.get("value"), dict) else {}
        effect = (val.get("effect") or j.get("effect") or "").strip()
        parts.append(f"{name} ({effect})" if effect else str(name))
    return "Jokers: " + ("; ".join(parts) if parts else "none")


def hand_brief(
    ante: int,
    blind_name: str,
    needed: int,
    hands_left: int,
    discards_left: int,
    hand: Sequence[Card],
    jokers: Sequence[Dict[str, Any]],
    options: Sequence[PlayOption],
    synergy_summary: str = "",
    notes: Sequence[str] = (),
) -> str:
    lines = [
        f"Balatro. Ante {ante}, {blind_name}.",
        f"Need {needed} more chips. {hands_left} hands and {discards_left} discards left.",
        _joker_line(jokers),
    ]
    if synergy_summary:
        lines.append(f"Your jokers reward: {synergy_summary}")
    lines.extend(notes)
    lines.append("Cards in hand: " + " ".join(c.short for c in hand))
    if options:
        lines.append("Hands you can make right now:")
        for opt in options:
            lines.append("- " + opt.label(needed, hands_left))
    return "\n".join(lines)


def discard_brief(
    ante: int,
    blind_name: str,
    needed: int,
    hands_left: int,
    discards_left: int,
    hand: Sequence[Card],
    plans: Sequence[DiscardPlan],
    synergy_summary: str = "",
    notes: Sequence[str] = (),
) -> str:
    lines = [
        f"Balatro. Ante {ante}, {blind_name}.",
        f"Need {needed} more chips. {hands_left} hands and {discards_left} discards left.",
        "Cards in hand: " + " ".join(c.short for c in hand),
    ]
    if synergy_summary:
        lines.append(f"Your jokers reward: {synergy_summary}")
    lines.extend(notes)
    lines.append("Discarding draws that many new cards from the deck.")
    return "\n".join(lines)


def shop_brief(
    ante: int,
    money: int,
    jokers: Sequence[Dict[str, Any]],
    joker_slots: str,
    synergy_summary: str = "",
    interest: int = 0,
) -> str:
    lines = [
        f"Balatro shop. Ante {ante}. You have ${money}.",
        f"Joker slots used: {joker_slots}.",
        _joker_line(jokers),
    ]
    if synergy_summary:
        lines.append(f"Your jokers reward: {synergy_summary}")
    lines.append(f"Money kept earns ${interest} interest next round (max $5 at $25).")
    return "\n".join(lines)


def blind_brief(
    ante: int,
    blind_name: str,
    target: int,
    money: int,
    jokers: Sequence[Dict[str, Any]],
    tag_name: str = "",
    tag_effect: str = "",
) -> str:
    lines = [
        f"Balatro blind choice. Ante {ante}, {blind_name} needs {target} chips.",
        f"You have ${money}.",
        _joker_line(jokers),
    ]
    if tag_name:
        lines.append(f"Skipping gives the tag {tag_name}" + (f": {tag_effect}" if tag_effect else ""))
        lines.append("Playing gives cash and a shop visit; skipping gives neither.")
    return "\n".join(lines)
