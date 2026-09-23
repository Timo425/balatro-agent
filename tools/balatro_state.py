#!/usr/bin/env python3
"""Loeb Balatro jooksva run'i save-failist ja prindib selle LLM-ile loetava tekstina.

Modi ega Balatrobotit pole vaja: mäng salvestab run'i ise faili save.jkr
(raw DEFLATE + Lua tabel). Ainult lugemine, mängu ei puudutata, achievementid jäävad alles.

Kasutus:
    python3 tools/balatro_state.py            # profiil 1
    python3 tools/balatro_state.py --profile 2
    python3 tools/balatro_state.py --json     # toores dump silumiseks

Nagu ülejäänud repo: ainult faktid, mitte soovitused (vt CLAUDE.md).
"""
import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

SAVE_ROOT = Path.home() / (
    ".steam/debian-installation/steamapps/compatdata/2379780/pfx/drive_c/"
    "users/steamuser/AppData/Roaming/Balatro"
)

STATES = {
    1: "SELECTING_HAND (in a blind, choosing cards)", 2: "HAND_PLAYED", 3: "DRAW_TO_HAND",
    4: "GAME_OVER", 5: "SHOP", 6: "PLAY_TAROT", 7: "BLIND_SELECT", 8: "ROUND_EVAL (cash out)",
    9: "TAROT_PACK", 10: "PLANET_PACK", 11: "MENU", 15: "SPECTRAL_PACK", 17: "STANDARD_PACK",
    18: "BUFFOON_PACK", 19: "NEW_ROUND",
}

# Bossi efektid, sest save'is on ainult võti
BOSSES = {
    "bl_hook": "The Hook: discards 2 random cards held in hand after every hand played",
    "bl_ox": "The Ox: playing your most played poker hand sets money to $0",
    "bl_house": "The House: first hand is drawn face down",
    "bl_wall": "The Wall: extra large blind (4x base)",
    "bl_wheel": "The Wheel: 1 in 7 cards drawn face down",
    "bl_arm": "The Arm: decreases level of played poker hand by 1",
    "bl_club": "The Club: all Club cards debuffed",
    "bl_fish": "The Fish: cards drawn face down after each hand played",
    "bl_psychic": "The Psychic: must play exactly 5 cards",
    "bl_goad": "The Goad: all Spade cards debuffed",
    "bl_water": "The Water: start with 0 discards",
    "bl_window": "The Window: all Diamond cards debuffed",
    "bl_manacle": "The Manacle: -1 hand size",
    "bl_eye": "The Eye: no repeat poker hand types this round",
    "bl_mouth": "The Mouth: only one poker hand type can be played this round",
    "bl_plant": "The Plant: all face cards debuffed",
    "bl_serpent": "The Serpent: after play or discard, always draw exactly 3 cards",
    "bl_pillar": "The Pillar: cards played previously this ante are debuffed",
    "bl_needle": "The Needle: only 1 hand (blind is 1x base)",
    "bl_head": "The Head: all Heart cards debuffed",
    "bl_tooth": "The Tooth: lose $1 per card played",
    "bl_flint": "The Flint: base chips and mult are halved",
    "bl_mark": "The Mark: all face cards drawn face down",
    "bl_final_acorn": "Amber Acorn (final): flips and shuffles all jokers",
    "bl_final_leaf": "Verdant Leaf (final): all cards debuffed until 1 joker is sold",
    "bl_final_vessel": "Violet Vessel (final): very large blind (6x base)",
    "bl_final_heart": "Crimson Heart (final): one random joker disabled every hand",
    "bl_final_bell": "Cerulean Bell (final): forces 1 card to always be selected",
}
BLIND_MULT = {"bl_small": 1, "bl_big": 1.5, "bl_wall": 4, "bl_needle": 1, "bl_final_vessel": 6}

# Blindi baas-chipid ante 1..8 järgi, indeks = GAME.modifiers.scaling (Green/Purple stake tõstavad)
ANTE_BASE = {
    1: [300, 800, 2000, 5000, 11000, 20000, 35000, 50000],
    2: [300, 900, 2600, 8000, 20000, 36000, 60000, 100000],
    3: [300, 1000, 3200, 9000, 25000, 60000, 110000, 200000],
}
STAKES = {1: "White", 2: "Red", 3: "Green", 4: "Black", 5: "Blue", 6: "Purple", 7: "Orange", 8: "Gold"}
RANK_ORDER = ["Ace", "King", "Queen", "Jack", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]


def parse_lua(text):
    """Balatro serialiseerib ainult kujul {["k"]=v,...}, seega piisab väikesest parserist."""
    pos = 0
    num = re.compile(r"-?(?:inf|nan|[\d.]+(?:[eE][+-]?\d+)?)")

    def skip():
        nonlocal pos
        while text[pos] in " \t\r\n,":
            pos += 1

    def value():
        nonlocal pos
        skip()
        c = text[pos]
        if c == "{":
            pos += 1
            out = {}
            while True:
                skip()
                if text[pos] == "}":
                    pos += 1
                    return out
                pos += 1  # [
                key = value()
                skip()
                pos += 2  # ]=
                out[key] = value()
        if c == '"':
            end = pos + 1
            while text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            s = text[pos + 1:end].replace('\\"', '"')
            pos = end + 1
            return s
        for word, v in (("true", True), ("false", False), ("nil", None)):
            if text.startswith(word, pos):
                pos += len(word)
                return v
        m = num.match(text, pos)
        pos = m.end()
        f = float(m.group())
        return int(f) if f.is_integer() else f

    return value()


def load(profile):
    path = SAVE_ROOT / str(profile) / "save.jkr"
    if not path.exists():
        sys.exit(f"Save'i pole: {path} (menüüs? run läbi?)")
    text = zlib.decompress(path.read_bytes(), -15).decode("utf-8", "replace")
    return path, parse_lua(text.removeprefix("return "))


def cards(area):
    c = (area or {}).get("cards", {})
    return [c[k] for k in sorted(c)]


def playing_card(c):
    b, ab = c["base"], c["ability"]
    s = f"{b['value']} of {b['suit']}"
    if ab["name"] not in ("Default Base", ""):
        s += f" [{ab['name']}]"
    if c.get("edition"):
        s += f" ({c['edition']['type']})"
    if c.get("seal"):
        s += f" <{c['seal']} seal>"
    if ab.get("perma_bonus"):
        s += f" +{ab['perma_bonus']} perma chips"
    if c.get("debuff"):
        s += " DEBUFFED"
    if c.get("facing") == "back" and c.get("area_hint") == "hand":
        s += " (face down)"
    return s


def joker_line(c, show_cost=False):
    ab = c["ability"]
    s = c.get("label") or ab.get("name")
    tags = []
    if c.get("edition"):
        tags.append(c["edition"]["type"])
    for st in ("eternal", "perishable", "rental"):
        if ab.get(st):
            tags.append(st + (f" {ab.get('perish_tally')} rounds left" if st == "perishable" else ""))
    if c.get("debuff"):
        tags.append("DEBUFFED")
    if tags:
        s += " (" + ", ".join(tags) + ")"
    # Kasvavad/muutuvad väärtused, mis pole kaardi nimest näha
    state = {}
    for k in ("mult", "x_mult", "chips", "t_mult", "t_chips"):
        v = ab.get(k)
        if v not in (None, 0, 1, 0.0) and ab.get("set") == "Joker":
            state[k] = v
    extra = ab.get("extra")
    if isinstance(extra, dict):
        state.update({k: v for k, v in extra.items() if not isinstance(v, dict)})
    elif extra not in (None, 0) and ab.get("set") == "Joker":
        state["extra"] = extra
    if state:
        s += " " + json.dumps(state)
    if show_cost:
        s += f" — ${c.get('cost')}"
    else:
        s += f" — sells ${c.get('sell_cost')}"
    return s


def blind_target(G, back, key):
    ante = G["round_resets"]["ante"]
    table = ANTE_BASE.get(G["modifiers"].get("scaling", 1), ANTE_BASE[1])
    base = table[min(max(ante, 1), 8) - 1] if ante >= 1 else 100
    base *= back.get("effect", {}).get("config", {}).get("ante_scaling", 1)
    return int(base * BLIND_MULT.get(key, 2))


def render(path, g):
    G, A, back = g["GAME"], g["cardAreas"], g["BACK"]
    cr, rr = G["current_round"], G["round_resets"]
    age = int(time.time() - path.stat().st_mtime)
    out = []
    p = out.append

    p(f"# Balatro run state (save written {age // 60}m {age % 60}s ago)")
    p(f"Deck: {back['name']} | Stake: {STAKES.get(G.get('stake'), G.get('stake'))} | "
      f"Ante {rr['ante']}/{G['win_ante']} | Round {G['round']}")
    p(f"Screen: {STATES.get(g['STATE'], g['STATE'])}")
    p(f"Money: ${G['dollars']} (interest ${G['interest_amount']} per $5, cap ${G['interest_cap']})")
    p(f"Hands left: {cr['hands_left']} | Discards left: {cr['discards_left']} | "
      f"Hand size: {A['hand']['config']['card_limit']} | "
      f"Joker slots: {len(cards(A['jokers']))}/{A['jokers']['config']['card_limit']} | "
      f"Consumable slots: {len(cards(A['consumeables']))}/{A['consumeables']['config']['card_limit']}")

    p("\n## Blinds this ante")
    for slot in ("Small", "Big", "Boss"):
        key = rr["blind_choices"][slot]
        desc = BOSSES.get(key, key.removeprefix("bl_").title() + " Blind")
        tag = rr.get("blind_tags", {}).get(slot)
        line = f"- {slot}: {desc} — needs {blind_target(G, back, key):,} chips — {rr['blind_states'][slot]}"
        if tag:
            line += f" (skip tag: {tag.removeprefix('tag_')})"
        p(line)
    blind = g.get("BLIND", {})
    if blind.get("name"):
        p(f"Current blind: {blind['name']} — score so far {G.get('chips', 0):,} / {blind.get('chips'):,}")

    p("\n## Jokers (left to right, order matters)")
    p("\n".join(f"- {joker_line(c)}" for c in cards(A["jokers"])) or "- none")

    p("\n## Consumables")
    p("\n".join(f"- {joker_line(c)}" for c in cards(A["consumeables"])) or "- none")

    hand = cards(A["hand"])
    if hand:
        p("\n## Hand")
        for c in hand:
            c["area_hint"] = "hand"
            p(f"- {playing_card(c)}")

    # Pakid/shop — kõik alad, mis parasjagu olemas on
    for area, title in (("pack_cards", "Open pack"), ("shop_jokers", "Shop cards"),
                        ("shop_booster", "Shop packs"), ("shop_vouchers", "Shop voucher")):
        cs = cards(A.get(area))
        if cs:
            p(f"\n## {title}")
            p("\n".join(f"- {joker_line(c, show_cost=True)}" for c in cs))
    if g["STATE"] == 5:
        p(f"Reroll cost: ${cr['reroll_cost']}")

    deck = cards(A["deck"])
    full = deck + hand + cards(A.get("discard")) + cards(A.get("play"))
    p(f"\n## Deck: {len(deck)} cards left to draw, {len(full)} total")
    for label, pile in (("Remaining draw pile", deck), ("Full deck", full)):
        if not pile:
            continue
        suits = Counter(c["base"]["suit"] for c in pile)
        ranks = Counter(c["base"]["value"] for c in pile)
        p(f"{label}: " + ", ".join(f"{s} {suits[s]}" for s in SUITS if suits[s]) + " | "
          + " ".join(f"{r}:{ranks[r]}" for r in RANK_ORDER if ranks[r]))
    special = [playing_card(c) for c in full
               if c["ability"]["name"] != "Default Base" or c.get("edition") or c.get("seal")
               or c["ability"].get("perma_bonus")]
    if special:
        p("Modified cards: " + "; ".join(special))

    p("\n## Poker hand levels (level / chips x mult / times played this run)")
    for name, h in sorted(G["hands"].items(), key=lambda kv: kv[1]["order"]):
        if h["visible"] and (h["level"] > 1 or h["played"]):
            p(f"- {name}: L{h['level']} {h['chips']}x{h['mult']} — played {h['played']}")
    p("(unlisted hands are level 1 and unplayed)")

    vouchers = [k.removeprefix("v_") for k in G.get("used_vouchers", {})]
    p(f"\nVouchers owned: {', '.join(vouchers) or 'none'}")
    tags = [t.get("key", "?").removeprefix("tag_") for t in (g.get("tags") or {}).values()]
    p(f"Tags held: {', '.join(tags) or 'none'}")
    p(f"Bosses already seen: {', '.join(k.removeprefix('bl_') for k, v in G.get('bosses_used', {}).items() if v) or 'none'}")
    rs = G.get("round_scores", {})
    p(f"Run stats: best hand {rs.get('hand', {}).get('amt', 0):,} chips, "
      f"{rs.get('cards_played', {}).get('amt', 0)} cards played, "
      f"{rs.get('cards_discarded', {}).get('amt', 0)} discarded, {G.get('skips', 0)} blinds skipped")
    p("\nNote: the save only updates when the game autosaves (screen changes). "
      "If this looks stale, ask the player for a screenshot.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=1, type=int)
    ap.add_argument("--json", action="store_true", help="toores save JSON-ina")
    a = ap.parse_args()
    path, g = load(a.profile)
    print(json.dumps(g, indent=1) if a.json else render(path, g))


if __name__ == "__main__":
    main()
