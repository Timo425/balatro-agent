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
from collections import Counter, defaultdict
from math import comb
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
ROUND_TARGETS = {"Mail-In Rebate": "mail_card", "The Idol": "idol_card",
                 "Ancient Joker": "ancient_card", "Castle": "castle_card"}
STAKES = {1: "White", 2: "Red", 3: "Green", 4: "Black", 5: "Blue", 6: "Purple", 7: "Orange", 8: "Gold"}
# Tagid, mis muudavad shopi jokerit (edition + tasuta). Mäng rakendab need eventina alles
# PÄRAST save_run()-i, seega save'is on jokerid veel tavalised ja tagid veel käes.
EDITION_TAGS = {"tag_negative": "negative", "tag_foil": "foil", "tag_holo": "holo",
                "tag_polychrome": "polychrome"}
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


def pending_shop_tags(g):
    """Ennusta, milliseid shopi jokereid edition-tagid kohe pärast save'i muudavad.

    Järgib mängu loogikat (create_card_for_shop -> store_joker_modify): iga editionita
    Joker saab esimese veel kasutamata edition-tagi, tag kulub ära.
    """
    if g["STATE"] != 5:
        return {}, set()
    free = [(i, t["key"]) for i, t in sorted((g.get("tags") or {}).items())
            if t.get("key") in EDITION_TAGS]
    out, used = {}, set()
    for c in cards(g["cardAreas"].get("shop_jokers")):
        if not free:
            break
        if c["ability"].get("set") == "Joker" and not c.get("edition"):
            i, key = free.pop(0)
            out[id(c)] = EDITION_TAGS[key]
            used.add(i)
    return out, used


def joker_line(c, show_cost=False, G=None, pending=None):
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
    # Igal roundil vahetuv sihtmärk on GAME.current_round all, mitte kaardil endal
    target = ROUND_TARGETS.get(ab.get("name"))
    if target and G:
        t = G.get("current_round", {}).get(target, {})
        state["target"] = " of ".join(str(t[k]) for k in ("rank", "suit") if t.get(k))
    if ab.get("to_do_poker_hand"):
        state["target"] = ab["to_do_poker_hand"]
    if state:
        s += " " + json.dumps(state)
    if pending:
        s += f" — $0 ({pending} + free from a {pending.title()} Tag; the save predates the tag, so it still says ${c.get('cost')})"
    elif show_cost:
        s += f" — ${c.get('cost')}"
    else:
        s += f" — sells ${c.get('sell_cost')}"
    return s


def card_suits(c, smeared):
    """Mis mastidena kaart flushis loeb: Wild = kõik, Stone = mitte ükski, Smeared liidab paarid."""
    name = c["ability"]["name"]
    if name == "Stone Card":
        return set()
    if name == "Wild Card":
        return set(SUITS)
    suit = c["base"]["suit"]
    if smeared:
        return {"Hearts", "Diamonds"} if suit in ("Hearts", "Diamonds") else {"Spades", "Clubs"}
    return {suit}


def flush_odds(hold, deck_suit, deck_size, hand_size, need, draws):
    """Täpne tõenäosus, et käes on `need` sama masti kaarti pärast iga järjestikust tõmmet.

    Iga tõmme = viska kuni 5 teise masti kaarti ära (discard või mängitud käsi) ja tõmba
    sama palju tagasi. Hüpergeomeetriline jaotus olekute (käes, pakis mastis, paki suurus) üle.
    """
    dist = {(hold, deck_suit, deck_size): 1.0}
    out = []
    for _ in range(draws):
        new = defaultdict(float)
        for (h, S, D), pr in dist.items():
            k = min(5, hand_size - h, D)
            if h >= need or k <= 0:
                new[(h, S, D)] += pr
                continue
            for x in range(min(k, S) + 1):
                q = comb(S, x) * comb(D - S, k - x) / comb(D, k)
                if q:
                    new[(h + x, S - x, D - k)] += pr * q
        dist = new
        out.append(sum(pr for (h, _, _), pr in dist.items() if h >= need))
    return out


def blind_spots(g):
    """Mida see snapshot EI näita. Mäng salvestab ainult kontrollpunktides (vt learned.md),
    seega ostud, müügid, kasutatud consumable'id ja paki sisu jõuavad save'i hiljem või üldse mitte."""
    action = g.get("ACTION") or {}
    if action.get("type") == "use_card":
        name = next((c.get("label") or c["ability"]["name"]
                     for area in g["cardAreas"].values() for c in cards(area)
                     if c.get("sort_id") == action.get("card")), "a pack")
        return [f"Saved at the moment {name} was opened. Its contents are NOT in the save — "
                "ask the player for a screenshot of the pack."]
    state = g["STATE"]
    if state == 5:
        return ["Saved on entering the shop (or the last reroll). Anything bought, sold or used "
                "since then is NOT reflected: shop cards may already be bought, money may be lower, "
                "jokers/consumables may have changed. Ask the player what they bought before advising."]
    if state in (1, 2, 3):
        return ["Saved at the start of this hand/after the last discard. Tarot/Spectral cards used "
                "since then are NOT reflected (hand, deck and consumables may differ)."]
    if state == 7:
        return ["Saved on reaching blind select. A pack from a skip tag opened here is NOT in the save."]
    return []


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
    for note in blind_spots(g):
        p(f"⚠ Not in this snapshot: {note}")
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
        left = blind.get("chips", 0) - G.get("chips", 0)
        if left > 0 and cr["hands_left"]:
            p(f"Still needed: {left:,} chips with {cr['hands_left']} hand{'s' * (cr['hands_left'] != 1)} left "
              f"(= {left // cr['hands_left']:,} per hand on average)")

    p("\n## Jokers (left to right, order matters)")
    p("\n".join(f"- {joker_line(c, G=G)}" for c in cards(A["jokers"])) or "- none")

    p("\n## Consumables")
    p("\n".join(f"- {joker_line(c)}" for c in cards(A["consumeables"])) or "- none")

    hand = cards(A["hand"])
    if hand:
        p("\n## Hand")
        for c in hand:
            c["area_hint"] = "hand"
            p(f"- {playing_card(c)}")

    # Pakid/shop — kõik alad, mis parasjagu olemas on
    pending, used_tags = pending_shop_tags(g)
    for area, title in (("pack_cards", "Open pack"), ("shop_jokers", "Shop cards"),
                        ("shop_booster", "Shop packs"), ("shop_vouchers", "Shop voucher")):
        cs = cards(A.get(area))
        if cs:
            p(f"\n## {title}")
            p("\n".join(f"- {joker_line(c, show_cost=True, G=G, pending=pending.get(id(c)))}" for c in cs))
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
    if hand and deck and g["STATE"] in (1, 2, 3):
        names = {c["ability"]["name"] for c in cards(A["jokers"]) if not c.get("debuff")}
        smeared, need = "Smeared Joker" in names, 4 if "Four Fingers" in names else 5
        draws = min(cr["discards_left"] + cr["hands_left"] - 1, 4)
        if draws > 0:
            p(f"\n## Flush odds (exact, from the draw pile above)")
            p(f"One 'draw' = throw away up to 5 cards that are not the suit — with a discard, or by "
              f"playing them as a hand, which spends that hand. You have {cr['discards_left']} discards "
              f"and {cr['hands_left']} hands; the last hand must be the flush itself. Chance of holding "
              f"{need} of the suit after each draw:")
            for suit in SUITS:
                hold = sum(suit in card_suits(c, smeared) for c in hand)
                in_deck = sum(suit in card_suits(c, smeared) for c in deck)
                if hold >= need:
                    p(f"- {suit}: already {hold} in hand")
                    continue
                odds = flush_odds(hold, in_deck, len(deck), len(hand), need, draws)
                p(f"- {suit}: hold {hold}, {in_deck} in {len(deck)} cards → "
                  + " · ".join(f"{i + 1} draw{'s' * bool(i)} {o:.0%}" for i, o in enumerate(odds)))
            p("(Wild cards count for every suit, Stone for none. Not modelled: The Serpent, "
              "cards drawn face down, debuffs — a debuffed card still counts toward the flush but scores nothing.)")

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
    tags = [t.get("key", "?").removeprefix("tag_") for i, t in sorted((g.get("tags") or {}).items())
            if i not in used_tags]
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
