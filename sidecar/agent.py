"""Balatro agent: turns the board into decisions for a model, and the model's
answers into moves.

THE ONE RULE OF THIS FILE (see CLAUDE.md for the long version):

    Python enumerates legal moves and states facts about them.
    The backend -- Laya, an LLM, or a human -- chooses which one happens.

So there is no `if best_hand_beats_target: play it` anywhere below, no sorting of
options by quality, no "top 3 only", and no silent heuristic when the model is
unsure. Where a decision is unavoidable and the model could not answer, the
fallback is counted in `backend.fallbacks` and labelled FALLBACK in the reasoning
so that a run tainted by Python's own play is visible in the telemetry.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sidecar.backends import MAX_OPTIONS_PER_QUESTION, BaseBackend, create_backend
from sidecar.brief import blind_brief, discard_brief, hand_brief, shop_brief
from sidecar.candidates import (
    DiscardPlan,
    PlayOption,
    annotate_play_options,
    enumerate_discard_plans,
    enumerate_play_cores,
    pick_filler,
)
from sidecar.poker_eval import Card
from sidecar.scoring import strength_label
from sidecar.strategy import StrategyManager
from sidecar.synergy import JokerSynergyAnalyzer, SynergyProfile


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
    action_type: str  # "PLAY" or "DISCARD"
    cards: List[Card]
    reasoning: str
    confidence: float


@dataclass
class HandFacts:
    """Everything true about the current hand, with nothing ranked or chosen."""

    needed: int
    plays: List[PlayOption]
    discards: List[DiscardPlan]
    jokers: List[Dict[str, Any]]
    profile: SynergyProfile
    play_context: str
    discard_context: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:24] or "item"


def _jokers_from(raw_g: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    val = (raw_g or {}).get("jokers")
    if isinstance(val, dict):
        cards = val.get("cards", [])
        return [c for c in cards if isinstance(c, dict)] if isinstance(cards, list) else []
    if isinstance(val, list):
        return [c for c in val if isinstance(c, dict)]
    return []


def _cost_of(item: Dict[str, Any], default: int) -> int:
    cost = item.get("cost")
    if isinstance(cost, dict):
        return int(cost.get("buy", default))
    return int(cost) if isinstance(cost, int) else default


def _sell_value(joker: Dict[str, Any]) -> int:
    cost = joker.get("cost")
    if isinstance(cost, dict):
        return int(cost.get("sell", 1))
    return int(cost) if isinstance(cost, int) else 1


def _effect_of(item: Dict[str, Any]) -> str:
    val = item.get("value") if isinstance(item.get("value"), dict) else {}
    return str(item.get("effect") or val.get("effect") or "").strip()


def _name_of(item: Dict[str, Any], fallback: str = "Item") -> str:
    return str(item.get("name") or item.get("label") or item.get("key") or fallback)


def _is_eternal(joker: Dict[str, Any]) -> bool:
    mod = joker.get("modifier")
    return bool(isinstance(mod, dict) and mod.get("eternal"))


class BalatroAgent:
    """Presents the board to a decision backend and executes what it answers."""

    def __init__(
        self,
        rules_path: str = "rules/balatro-strategy.md",
        strategy_name: str = "default",
        backend: Any = "laya",
        model: Optional[str] = None,
        tree_mode: Optional[bool] = None,
        poll_mode: Optional[bool] = None,
        **backend_kwargs,
    ):
        self.rules_path = Path(rules_path)
        self.strategy_name = strategy_name
        self.sm = StrategyManager(strategy_name)
        if isinstance(backend, BaseBackend):
            self.backend = backend
        else:
            self.backend = create_backend(backend_type=str(backend), model=model, **backend_kwargs)
        self.agent = getattr(self.backend, "agent", None)
        if tree_mode is None:
            tree_mode = bool(getattr(self.backend, "is_tree", False)) or str(backend).lower() in (
                "laya_tree", "tree", "hierarchical", "api_llm_tree", "llm_tree",
            )
        self.tree_mode = tree_mode
        self.poll_mode = bool(poll_mode) if poll_mode is not None else self.tree_mode
        self.analyzer = JokerSynergyAnalyzer()

    # ------------------------------------------------------------------
    # Backend plumbing
    # ------------------------------------------------------------------

    def _get_max_chars(self) -> int:
        return getattr(self.backend, "max_chars", 4000)

    def _context(self, brief: str, raw_g: Optional[Dict[str, Any]]) -> str:
        """The brief for decision models; the full markdown gamestate for LLMs.

        Laya reads a few hundred tokens and truncates the rest, so handing it the
        full BalatroLLM markdown cut off the hand itself. LLM backends have room
        for everything, and lose nothing by getting both.
        """
        if getattr(self.backend, "wants_full_context", False) and raw_g:
            try:
                return (brief + "\n\n" + self.sm.render_gamestate(raw_g))[: self._get_max_chars()]
            except Exception:
                pass
        return brief[: self._get_max_chars()]

    def _choose(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> Optional[str]:
        """Asks the backend to pick a key. Returns None when it could not answer."""
        if not criteria:
            return None
        try:
            choice = self.backend.predict_choice(context, question_id, instructions, criteria)
        except Exception:
            choice = None
        if choice in criteria:
            return choice
        self.backend.note_fallback()
        return None

    def _poll(
        self,
        context: str,
        question_id: str,
        statement_for,
        criteria: Dict[str, str],
    ) -> Optional[str]:
        """Asks the backend to score every option, and returns its highest.

        Used where the option list is long or the model answers yes/no questions
        better than wide multiple choice. Python compares nothing of its own here:
        every score is the model's.
        """
        if not criteria:
            return None
        try:
            scores = self.backend.rank_options(context, question_id, statement_for, criteria)
        except Exception:
            scores = {}
        scores = {k: v for k, v in scores.items() if k in criteria}
        if not scores:
            self.backend.note_fallback()
            return None
        return max(scores, key=lambda k: scores[k])

    def _pick(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
        statement_for=None,
    ) -> Optional[str]:
        """One decision node: a poll when polling is enabled, a choice otherwise."""
        if self.poll_mode and statement_for is not None:
            return self._poll(context, question_id, statement_for, criteria)
        return self._choose(context, question_id, instructions, criteria)

    def _ask_bool(
        self,
        context: str,
        question_id: str,
        statement: str,
        true_desc: str,
        false_desc: str,
        on_failure: bool,
    ) -> bool:
        try:
            return bool(self.backend.predict_bool(context, question_id, statement, true_desc, false_desc))
        except Exception:
            self.backend.note_fallback()
            return on_failure

    # ------------------------------------------------------------------
    # Hand decisions
    # ------------------------------------------------------------------

    def _hand_facts(
        self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None
    ) -> HandFacts:
        hands_data = (raw_g or {}).get("hands", {})
        jokers = _jokers_from(raw_g)
        profile = self.analyzer.analyze(jokers)
        needed = max(0, state.target_chips - state.current_chips)

        plays = annotate_play_options(
            enumerate_play_cores(state.hand),
            hands_data,
            jokers,
            hands_left=state.hands_left,
            discards_left=state.discards_left,
        )
        discards: List[DiscardPlan] = []
        if state.discards_left > 0:
            discards = enumerate_discard_plans(
                state.hand,
                favored_suits=profile.favored_suits,
                favored_face_cards=profile.favored_face_cards,
            )

        summary = profile.summary if profile.has_synergies else ""
        notes = self._resource_notes(profile)
        play_context = self._context(
            hand_brief(
                state.ante, state.blind_name, needed, state.hands_left, state.discards_left,
                state.hand, jokers, plays, summary, notes,
            ),
            raw_g,
        )
        discard_context = self._context(
            discard_brief(
                state.ante, state.blind_name, needed, state.hands_left, state.discards_left,
                state.hand, discards, summary, notes,
            ),
            raw_g,
        )
        return HandFacts(needed, plays, discards, jokers, profile, play_context, discard_context)

    @staticmethod
    def _resource_notes(profile: SynergyProfile) -> List[str]:
        """Board-level consequences the joker text implies, stated as facts.

        A joker that pays for each unused discard makes discarding cost score --
        true of the whole turn rather than of any one option, so it belongs in the
        brief. Whether that is worth it is still the model's call.
        """
        notes = []
        if "discard" in profile.resource_bonuses:
            notes.append("Your jokers pay for each unused discard, so discarding lowers your score.")
        if "hand" in profile.resource_bonuses:
            notes.append("Your jokers pay for each unused hand.")
        return notes

    def _play_criteria(self, facts: HandFacts, state: BalatroGameState,
                       options: Sequence[PlayOption]) -> Dict[str, str]:
        criteria = {}
        for opt in options:
            label = opt.label(facts.needed, state.hands_left)
            synergy = self.analyzer.annotate_hand(facts.profile, opt.core, opt.hand_type)
            criteria[opt.key] = f"{label}; {synergy}" if synergy else label
        return criteria

    def decide_hand_action(
        self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None
    ) -> AgentAction:
        if not state.hand:
            return AgentAction("PLAY", [], "No cards in hand", 0.0)

        facts = self._hand_facts(state, raw_g)
        if not facts.plays:
            return AgentAction("PLAY", [], "No legal play", 0.0)

        if self.tree_mode:
            return self._hand_tree(state, facts)
        return self._hand_flat(state, facts)

    def _hand_flat(self, state: BalatroGameState, facts: HandFacts) -> AgentAction:
        """One question holding every legal move.

        This is the baseline the tree is measured against. It deliberately does not
        trim the option list: the cardinality is the thing being measured, and a
        decision model's accuracy falling away as options multiply is a result, not
        a bug to hide.
        """
        criteria = self._play_criteria(facts, state, facts.plays)
        by_key: Dict[str, Any] = {opt.key: opt for opt in facts.plays}
        for plan in facts.discards:
            key = f"discard_{plan.key}"
            criteria[key] = plan.label(state.discards_left)
            by_key[key] = plan

        choice = self._choose(
            facts.play_context,
            "hand_action",
            "Pick the move that best gets the player to the chips this blind needs.",
            criteria,
        )
        if choice is None:
            return self._fallback_play(facts)

        target = by_key[choice]
        if isinstance(target, DiscardPlan):
            return AgentAction("DISCARD", list(target.discard),
                               f"Decision: discard — {target.label(state.discards_left)}", 0.9)
        return AgentAction("PLAY", list(target.cards),
                           f"Decision: play — {target.label(facts.needed, state.hands_left)}", 0.9)

    def _hand_tree(self, state: BalatroGameState, facts: HandFacts) -> AgentAction:
        """Hierarchical version: a few small questions instead of one wide one.

        Node 1  dig?         yes/no -- discard and draw, or play a hand now
        Node 2a which hand?  one question when the options fit, otherwise
                             hand type first, then which hand of that type
        Node 2b which plan?  one of the named discard patterns
        Node 3  send filler? yes/no -- only when the chosen hand is under five cards
        """
        if facts.discards and state.discards_left > 0:
            # The verdicts go in the statement, not only in the brief: a question's
            # instructions are never truncated once the options are short, while
            # the state is cut from the end. Listing them in enumeration order is
            # an inventory of what exists, not a ranking of it.
            verdicts = ", ".join(
                strength_label(o.estimate.total, facts.needed)
                for o in facts.plays if o.estimate is not None
            )
            dig = self._ask_bool(
                facts.play_context,
                "dig_for_better_hand",
                (
                    f"The hands available right now are rated: {verdicts}. "
                    f"{state.hands_left} hands and {state.discards_left} discards are left. "
                    "Should the player throw cards away and draw new ones instead of "
                    "playing one of those hands?"
                ),
                "throw cards away now and draw replacements",
                "play one of the hands listed above now",
                on_failure=False,
            )
            if dig:
                return self._tree_discard(state, facts)

        return self._tree_play(state, facts)

    def _tree_play(self, state: BalatroGameState, facts: HandFacts) -> AgentAction:
        options = facts.plays
        if len(options) == 1:
            return self._finish_play(state, facts, options[0], "only legal hand")

        chosen: Optional[PlayOption]
        if len(options) <= MAX_OPTIONS_PER_QUESTION:
            criteria = self._play_criteria(facts, state, options)
            key = self._pick(
                facts.play_context, "which_hand",
                "Pick the hand that scores the most chips towards beating this blind.", criteria,
                statement_for=lambda _key, label: f"The player should play {label}",
            )
            chosen = next((o for o in options if o.key == key), None)
        else:
            chosen = self._narrow_by_hand_type(state, facts, options)

        if chosen is None:
            return self._fallback_play(facts)
        return self._finish_play(state, facts, chosen, "chosen")

    def _narrow_by_hand_type(
        self, state: BalatroGameState, facts: HandFacts, options: Sequence[PlayOption]
    ) -> Optional[PlayOption]:
        """Two-step coarse-to-fine choice, the split the Laya model card recommends.

        The grouping is by poker hand type -- a property of the cards, not a
        judgement about which group is better.
        """
        groups: Dict[str, List[PlayOption]] = {}
        for opt in options:
            groups.setdefault(opt.hand_type, []).append(opt)

        type_criteria = {}
        for hand_type, members in groups.items():
            key = _slug(hand_type)
            sample = members[0].label(facts.needed, state.hands_left)
            extra = f" ({len(members)} to choose from)" if len(members) > 1 else ""
            type_criteria[key] = f"{sample}{extra}"

        chosen_type = self._choose(facts.play_context, "which_hand_type",
                                   "Pick the kind of poker hand that scores the most chips here.",
                                   type_criteria)
        if chosen_type is None:
            return None

        members = next((v for k, v in groups.items() if _slug(k) == chosen_type), [])
        if not members:
            return None
        if len(members) == 1:
            return members[0]

        criteria = self._play_criteria(facts, state, members)
        key = self._pick(
            facts.play_context, "which_hand_of_type",
            "Pick the hand that scores the most chips towards beating this blind.", criteria,
            statement_for=lambda _key, label: f"The player should play {label}",
        )
        return next((o for o in members if o.key == key), None)

    def _finish_play(
        self, state: BalatroGameState, facts: HandFacts, option: PlayOption, why: str
    ) -> AgentAction:
        """Optionally attaches non-scoring filler, then returns the move."""
        spare = pick_filler(state.hand, option.core)
        if spare and len(option.core) < 5:
            cycle = self._ask_bool(
                facts.play_context,
                "send_filler_cards",
                (
                    f"The player will play {option.hand_type} "
                    f"({' '.join(c.short for c in option.core)}). Should they also play "
                    f"{len(spare)} spare cards ({' '.join(c.short for c in spare)}) that score "
                    "nothing, to get them out of hand and draw replacements?"
                ),
                "yes, send the spare cards away too",
                "no, play only the scoring cards and keep the rest",
                on_failure=False,
            )
            if cycle:
                option.filler = spare

        label = option.label(facts.needed, state.hands_left)
        return AgentAction("PLAY", list(option.cards), f"Tree ({why}): play {label}", 0.95)

    def _tree_discard(self, state: BalatroGameState, facts: HandFacts) -> AgentAction:
        plans = facts.discards
        if len(plans) == 1:
            plan = plans[0]
            return AgentAction("DISCARD", list(plan.discard),
                               f"Tree (only plan): {plan.label(state.discards_left)}", 0.95)

        criteria = {p.key: p.label(state.discards_left) for p in plans}
        key = self._pick(
            facts.discard_context, "which_discard",
            "Pick the cards to throw away so the next draw makes a stronger hand.", criteria,
            statement_for=lambda _key, label: f"The player should {label}",
        )
        plan = next((p for p in plans if p.key == key), None)
        if plan is None:
            fallback = plans[0]
            return AgentAction("DISCARD", list(fallback.discard),
                               f"FALLBACK (model gave no answer): {fallback.label(state.discards_left)}", 0.5)
        return AgentAction("DISCARD", list(plan.discard),
                           f"Tree: {plan.label(state.discards_left)}", 0.95)

    def _fallback_play(self, facts: HandFacts) -> AgentAction:
        """Used only when the backend could not answer at all.

        Plays the first enumerated option. Enumeration order is the game's fixed
        poker-hand ranking, so this is Python playing, not the model -- which is
        exactly why it is labelled and counted.
        """
        option = facts.plays[0]
        return AgentAction("PLAY", list(option.cards),
                           f"FALLBACK (model gave no answer): play {option.hand_type}", 0.5)

    # ------------------------------------------------------------------
    # Shop decisions
    # ------------------------------------------------------------------

    def decide_shop_action(
        self,
        state: BalatroGameState,
        shop_items: List[Dict[str, Any]],
        max_jokers: int = 5,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, int, str]]:
        """Returns ("BUY"|"SELL", index, reasoning), or None to leave the shop."""
        if state.money < 1 or not shop_items:
            return None

        jokers = _jokers_from(raw_g)
        profile = self.analyzer.analyze(jokers)
        slots = f"{len(jokers)}/{max_jokers}"
        interest = min(state.money // 5, 5)
        context = self._context(
            shop_brief(state.ante, state.money, jokers, slots,
                       profile.summary if profile.has_synergies else "", interest),
            raw_g,
        )

        groups = self._group_shop_items(shop_items, state.money)
        affordable = [entry for bucket in groups.values() for entry in bucket]
        if not affordable:
            return None

        if self.tree_mode:
            picked = self._shop_tree(context, state, groups, jokers, max_jokers, profile, interest)
        else:
            picked = self._shop_flat(context, state, affordable, profile, interest)

        if picked is None:
            return None
        idx, item = picked
        return self._buy_or_make_room(context, state, idx, item, jokers, max_jokers, profile)

    def _group_shop_items(
        self, shop_items: List[Dict[str, Any]], money: int
    ) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {
            "joker": [], "pack": [], "voucher": [], "consumable": [],
        }
        for idx, item in enumerate(shop_items):
            if _cost_of(item, 4) > money:
                continue
            name = _name_of(item).lower()
            set_type = str(item.get("set") or "").upper()
            if "voucher" in name or set_type == "VOUCHER":
                groups["voucher"].append((idx, item))
            elif "pack" in name or "booster" in name or set_type == "BOOSTER":
                groups["pack"].append((idx, item))
            elif any(w in name for w in ("planet", "tarot", "spectral")) or set_type in ("PLANET", "TAROT", "SPECTRAL"):
                groups["consumable"].append((idx, item))
            else:
                groups["joker"].append((idx, item))
        return groups

    def _shop_item_label(self, item: Dict[str, Any], money: int, profile: SynergyProfile) -> str:
        cost = _cost_of(item, 4)
        name = _name_of(item)
        effect = _effect_of(item)
        share = cost * 100 // money if money > 0 else 100
        budget = "cheap" if share <= 40 else ("most of your money" if share >= 80 else "affordable")
        synergy = self.analyzer.annotate_shop_item(profile, effect, name)
        label = f"{name} ${cost} ({budget})"
        if effect:
            label += f": {effect}"
        return label + synergy

    def _shop_flat(
        self,
        context: str,
        state: BalatroGameState,
        entries: List[Tuple[int, Dict[str, Any]]],
        profile: SynergyProfile,
        interest: int,
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        criteria = {
            f"buy{idx}_{_slug(_name_of(item))}": self._shop_item_label(item, state.money, profile)
            for idx, item in entries
        }
        criteria["leave_shop"] = f"buy nothing and keep ${state.money} (earns ${interest} interest)"
        choice = self._choose(context, "shop_action", "Pick what to buy to make the deck score more chips in later rounds.", criteria)
        if choice is None or choice == "leave_shop":
            return None
        return self._entry_for_key(choice, entries)

    def _shop_tree(
        self,
        context: str,
        state: BalatroGameState,
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]],
        jokers: List[Dict[str, Any]],
        max_jokers: int,
        profile: SynergyProfile,
        interest: int,
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        categories: Dict[str, str] = {}
        sellable = any(not _is_eternal(j) for j in jokers)
        for name, entries in groups.items():
            if not entries:
                continue
            if name == "joker" and len(jokers) >= max_jokers and not sellable:
                continue
            cheapest = min(_cost_of(it, 4) for _, it in entries)
            label = {
                "joker": f"a Joker ({len(entries)} on offer, from ${cheapest}, slots {len(jokers)}/{max_jokers})",
                "pack": f"a Booster Pack ({len(entries)} on offer, from ${cheapest})",
                "voucher": f"a permanent Voucher upgrade ({len(entries)} on offer, from ${cheapest})",
                "consumable": f"a Tarot or Planet card ({len(entries)} on offer, from ${cheapest})",
            }[name]
            categories[f"cat_{name}"] = label
        if not categories:
            return None
        categories["cat_save"] = f"nothing, keep ${state.money} (earns ${interest} interest)"

        chosen = self._choose(context, "shop_category", "Pick what to spend money on to score more chips in later rounds.", categories)
        if chosen is None or chosen == "cat_save":
            return None

        entries = groups.get(chosen.replace("cat_", ""), [])
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0]

        criteria = {
            f"buy{idx}_{_slug(_name_of(item))}": self._shop_item_label(item, state.money, profile)
            for idx, item in entries
        }
        pick = self._choose(context, "which_shop_item", "Pick the one that makes the deck score more chips in later rounds.", criteria)
        if pick is None:
            return None
        return self._entry_for_key(pick, entries)

    @staticmethod
    def _entry_for_key(
        key: str, entries: List[Tuple[int, Dict[str, Any]]]
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        match = re.match(r"buy(\d+)_", key)
        if not match:
            return None
        wanted = int(match.group(1))
        return next(((i, it) for i, it in entries if i == wanted), None)

    def _buy_or_make_room(
        self,
        context: str,
        state: BalatroGameState,
        idx: int,
        item: Dict[str, Any],
        jokers: List[Dict[str, Any]],
        max_jokers: int,
        profile: SynergyProfile,
    ) -> Optional[Tuple[str, int, str]]:
        """Buys the chosen item, asking which joker to sell first if slots are full."""
        name = _name_of(item)
        set_type = str(item.get("set") or "").upper()
        looks_like_joker = not any(
            w in name.lower() for w in ("voucher", "pack", "booster", "planet", "tarot", "spectral")
        ) and set_type not in ("VOUCHER", "BOOSTER", "PLANET", "TAROT", "SPECTRAL")

        if not looks_like_joker or len(jokers) < max_jokers:
            return ("BUY", idx, f"Decision: buy '{name}'")

        criteria = {}
        for j_idx, joker in enumerate(jokers):
            if _is_eternal(joker):
                continue
            j_name = _name_of(joker, f"Joker #{j_idx}")
            lost = self.analyzer.joker_contribution(profile, j_name, _effect_of(joker))
            criteria[f"sell{j_idx}_{_slug(j_name)}"] = (
                f"sell {j_name} for ${_sell_value(joker)}, losing {lost}"
            )
        if not criteria:
            return None
        criteria["keep_all_jokers"] = f"keep all jokers and do not buy '{name}'"

        choice = self._choose(
            context, "sell_joker_for_room",
            f"Joker slots are full. Which joker should go to make room for '{name}'?",
            criteria,
        )
        if choice is None or choice == "keep_all_jokers":
            return None
        match = re.match(r"sell(\d+)_", choice)
        if not match:
            return None
        s_idx = int(match.group(1))
        s_name = _name_of(jokers[s_idx], f"Joker #{s_idx}")
        return ("SELL", s_idx, f"Decision: sell '{s_name}' to make room for '{name}'")

    # ------------------------------------------------------------------
    # Blind, pack, consumable and joker-order decisions
    # ------------------------------------------------------------------

    def decide_blind_action(
        self, state: BalatroGameState, raw_g: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        if not raw_g:
            return ("SELECT", "Playing blind.")

        blinds = raw_g.get("blinds", {}) if isinstance(raw_g.get("blinds"), dict) else {}
        info, kind = None, None
        for key in ("small", "big", "boss"):
            b = blinds.get(key)
            if not isinstance(b, dict):
                continue
            status = str(b.get("status", "")).upper()
            if status in ("SELECT", "SELECTING", "CURRENT", "ACTIVE") or b.get("selected"):
                info, kind = b, key.upper()
                break
        if info is None:
            for key in ("small", "big", "boss"):
                b = blinds.get(key)
                if isinstance(b, dict) and str(b.get("status", "")).upper() not in ("DEFEATED", "SKIPPED"):
                    info, kind = b, key.upper()
                    break
        if info is None:
            return ("SELECT", "Playing blind (blind info unavailable).")

        kind = kind or "SMALL"
        if kind == "BOSS":
            boss = info.get("name") or "Boss Blind"
            return ("SELECT", f"Playing {boss} (Boss blinds cannot be skipped).")

        blind_name = info.get("name") or f"{kind.capitalize()} Blind"
        tag_name = info.get("tag_name") or info.get("tag") or ""
        tag_effect = info.get("tag_effect") or ""
        if not tag_name:
            return ("SELECT", f"Playing {blind_name} (no tag offered for skipping).")

        jokers = _jokers_from(raw_g)
        context = self._context(
            blind_brief(state.ante, blind_name,
                        info.get("chips") or info.get("score") or state.target_chips,
                        state.money, jokers, tag_name, tag_effect),
            raw_g,
        )
        skip = self._ask_bool(
            context,
            "skip_blind",
            f"Should the player skip {blind_name} to take the tag '{tag_name}' instead of playing for cash?",
            f"skip and take '{tag_name}'",
            f"play {blind_name} for the cash payout and the shop",
            on_failure=False,
        )
        if skip:
            return ("SKIP", f"Decision: skip {blind_name} for tag '{tag_name}'")
        return ("SELECT", f"Decision: play {blind_name} for cash and shop.")

    def decide_pack_card(
        self,
        pack_cards: List[Dict[str, Any]],
        hand: Optional[List[Card]] = None,
        jokers_count: int = 0,
        max_jokers: int = 5,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, Optional[List[int]], str]]:
        if not pack_cards:
            return None

        jokers = _jokers_from(raw_g)
        profile = self.analyzer.analyze(jokers)
        criteria = {}
        for idx, card in enumerate(pack_cards):
            name = _name_of(card, f"Card #{idx}")
            effect = _effect_of(card)
            set_type = str(card.get("set") or "CARD").title()
            synergy = self.analyzer.annotate_shop_item(profile, effect, name)
            label = f"{set_type} {name}" + (f": {effect}" if effect else "")
            criteria[f"take{idx}_{_slug(name)}"] = label + synergy
        criteria["skip_pack"] = "take nothing from this pack"

        context = self._context(
            "Balatro booster pack. " + ("Jokers: " + ", ".join(
                _name_of(j) for j in jokers) if jokers else "Jokers: none"),
            raw_g,
        )
        choice = self._choose(context, "pack_choice", "Which card should the player take?", criteria)
        if choice is None or choice == "skip_pack":
            return None

        match = re.match(r"take(\d+)_", choice)
        if not match:
            return None
        idx = int(match.group(1))
        card = pack_cards[idx]
        name = _name_of(card, "card")
        targets = None
        if self.consumable_requires_hand(card) and hand:
            targets = self.decide_tarot_targets(name, hand=hand, raw_g=raw_g, effect=_effect_of(card))
        return (idx, targets, f"Decision: take '{name}' from pack")

    @staticmethod
    def consumable_requires_hand(c: Dict[str, Any]) -> bool:
        """Whether a consumable needs cards highlighted in hand before it can be used."""
        set_type = (c.get("set") or "").upper()
        if set_type == "PLANET":
            return False

        name = (c.get("label") or c.get("key") or "").lower()
        non_hand_names = {
            "hermit", "the hermit", "fool", "the fool", "temperance",
            "judgement", "the judgement", "wheel of fortune", "the wheel of fortune",
            "emperor", "the emperor", "high priestess", "the high priestess",
        }
        if any(nh in name for nh in non_hand_names):
            return False

        effect = _effect_of(c).lower()
        if effect:
            if re.search(r"select(?:ed)?\s+(\d+\s+)?cards?|\bconvert\b|\bdestroy\b|\benhance\b", effect):
                return True
            if any(k in effect for k in ["money", "cash", "joker", "copy", "edition", "creates up to"]):
                return False

        return set_type == "TAROT"

    def decide_tarot_targets(
        self,
        tarot_name: str,
        hand: Optional[List[Any]] = None,
        raw_g: Optional[Dict[str, Any]] = None,
        effect: str = "",
    ) -> Optional[List[int]]:
        """Asks which card(s) in hand a Tarot should be applied to.

        How many targets the card takes is read from its own effect text, so no
        hardcoded list of Tarot names is needed.
        """
        if not hand:
            return None

        eff = (effect or "").lower()
        if eff:
            match = re.search(
                r"(?:select(?:ed)?\s+(?:up\s+to\s+)?(\d+)\s+cards?|(\d+)\s+(?:up\s+to\s+)?(?:selected\s+)?cards?)",
                eff,
            )
            if match:
                max_targets = int(match.group(1) or match.group(2))
            elif "selected card" in eff or "select 1 card" in eff:
                max_targets = 1
            elif any(k in eff for k in ("convert", "destroy", "enhance", "enhances")):
                max_targets = 2 if ("2" in eff or "two" in eff) else 1
            else:
                return None
        else:
            if any(k in tarot_name.lower() for k in ("money", "cash", "joker", "copy", "edition")):
                return None
            max_targets = 1

        criteria = {
            f"card{idx}_{_slug(getattr(c, 'rank', '?'))}": (
                f"the {getattr(c, 'rank', '?')} of {getattr(c, 'suit', '?')}"
            )
            for idx, c in enumerate(hand)
        }
        if not criteria:
            return None

        context = self._context(
            f"Balatro. Applying '{tarot_name}'" + (f" ({effect})" if effect else "")
            + ". Cards in hand: " + " ".join(getattr(c, "short", "?") for c in hand),
            raw_g,
        )

        picked: List[int] = []
        remaining = dict(criteria)
        for slot in range(max(1, max_targets)):
            if not remaining:
                break
            choice = self._choose(
                context,
                f"tarot_target_{slot + 1}",
                f"Which card should '{tarot_name}' be used on?",
                remaining,
            )
            if choice is None:
                break
            match = re.match(r"card(\d+)_", choice)
            if not match:
                break
            picked.append(int(match.group(1)))
            remaining.pop(choice, None)

        return picked or None

    def decide_consumable_action(
        self,
        consumables: List[Dict[str, Any]],
        hand: Optional[List[Any]] = None,
        raw_g: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[int, Optional[List[int]], str]]:
        if not consumables:
            return None

        is_selecting_hand = ((raw_g or {}).get("state") or "") == "SELECTING_HAND"
        criteria = {}
        for idx, card in enumerate(consumables):
            if not is_selecting_hand and self.consumable_requires_hand(card):
                continue
            name = _name_of(card, f"Consumable #{idx}")
            effect = _effect_of(card)
            set_type = str(card.get("set") or "CONSUMABLE").title()
            criteria[f"use{idx}_{_slug(name)}"] = f"use {set_type} {name}" + (f": {effect}" if effect else "")
        if not criteria:
            return None
        criteria["keep_all"] = "keep them for later"

        jokers = _jokers_from(raw_g)
        context = self._context(
            "Balatro. Deciding whether to use a consumable now.\n"
            + ("Jokers: " + ", ".join(_name_of(j) for j in jokers) if jokers else "Jokers: none"),
            raw_g,
        )
        choice = self._choose(context, "consumable_action",
                              "Should the player use a consumable now, and which one?", criteria)
        if choice is None or choice == "keep_all":
            return None

        match = re.match(r"use(\d+)_", choice)
        if not match:
            return None
        idx = int(match.group(1))
        card = consumables[idx]
        name = _name_of(card, "consumable")
        targets = self.decide_tarot_targets(name, hand=hand, raw_g=raw_g, effect=_effect_of(card))
        return (idx, targets, f"Decision: use '{name}' (targets={targets})")

    def decide_joker_order(
        self, current_jokers: List[Dict[str, Any]], raw_g: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        """Asks whether to reorder jokers so X-Mult triggers last.

        Balatro scores jokers left to right, so a flat +Mult applied after an
        X-Mult is worth less. Python works out what that order would be; the model
        decides whether to adopt it.
        """
        if len(current_jokers) <= 1:
            return None

        phases = []
        for i, joker in enumerate(current_jokers):
            effect = _effect_of(joker).lower()
            if re.search(r"(?<![a-z])x\s*\d+(\.\d+)?|\bx\s*mult\b|\bxmult\b", effect):
                phase = 2
            elif re.search(r"\b\+\s*\d+\s*mult\b|\bmult\b", effect):
                phase = 1
            else:
                phase = 0
            phases.append((phase, i))

        candidate = [i for _, i in sorted(phases, key=lambda x: x[0])]
        if candidate == list(range(len(current_jokers))):
            return None

        names = [_name_of(j, f"Joker #{i}") for i, j in enumerate(current_jokers)]
        criteria = {
            "order_current": "keep them as they are: " + ", ".join(names),
            "order_xmult_last": "move to: " + ", ".join(names[i] for i in candidate)
            + " (chips and +Mult first, X-Mult last)",
        }
        context = self._context(
            "Balatro scores jokers left to right. Jokers now: "
            + "; ".join(f"{n} ({_effect_of(j)})" for n, j in zip(names, current_jokers)),
            raw_g,
        )
        choice = self._choose(context, "joker_order", "How should the jokers be ordered?", criteria)
        return candidate if choice == "order_xmult_last" else None
