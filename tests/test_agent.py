"""Agent behaviour with a scripted model, plus the prompt-budget guard.

These tests drive the agent with a ScriptedBackend rather than the real Laya
model: what is under test is that the agent presents the right options and
carries out whatever it is told, not how well any particular model plays.
"""

import unittest

from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.backends import head_budget
from sidecar.poker_eval import Card
from tests.support import ScriptedBackend

# Both Laya checkpoints: the English root and typed-decisions.
HEAD_BUDGETS = {"laya (English root)": 192, "laya-typed-decisions": 256}


def spades_flush_hand():
    return [Card("A", "Spades", 0), Card("K", "Spades", 1), Card("Q", "Spades", 2),
            Card("J", "Spades", 3), Card("9", "Spades", 4), Card("2", "Hearts", 5),
            Card("3", "Clubs", 6), Card("4", "Diamonds", 7)]


def four_hearts_hand():
    return [Card("K", "Hearts", 0), Card("10", "Hearts", 1), Card("7", "Hearts", 2),
            Card("4", "Hearts", 3), Card("2", "Spades", 4), Card("Q", "Clubs", 5),
            Card("9", "Diamonds", 6), Card("3", "Spades", 7)]


def state(hand, **overrides):
    base = dict(ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
                hands_left=4, discards_left=3, money=4, hand=hand, jokers=[])
    base.update(overrides)
    return BalatroGameState(**base)


class TestOptionsOffered(unittest.TestCase):
    def test_a_winning_flush_is_offered_and_labelled_as_a_win(self):
        backend = ScriptedBackend({"dig_for_better_hand": False, "send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(state(spades_flush_hand()), raw_g={"hands": {}})
        criteria = backend.asked("which_hand")[0]
        flush = next(v for k, v in criteria.items() if k.startswith("flush_"))
        self.assertIn("WINS", flush)

    def test_flush_draw_is_offered_when_holding_four_of_a_suit(self):
        backend = ScriptedBackend({"dig_for_better_hand": True})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        action = agent.decide_hand_action(state(four_hearts_hand(), target_chips=800),
                                          raw_g={"hands": {}})
        self.assertEqual(action.action_type, "DISCARD")
        criteria = backend.asked("which_discard")[0]
        self.assertIn("flush_draw_hearts", criteria)

    def test_no_dig_question_without_discards(self):
        backend = ScriptedBackend({"send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(state(spades_flush_hand(), discards_left=0),
                                 raw_g={"hands": {}})
        self.assertNotIn("dig_for_better_hand", backend.question_ids)

    def test_empty_hand_is_handled(self):
        agent = BalatroAgent(backend=ScriptedBackend(), tree_mode=True)
        self.assertEqual(agent.decide_hand_action(state([])).cards, [])


class TestPromptBudget(unittest.TestCase):
    """Laya charges options against the head budget before the instructions.

    When they overrun it, the instructions are cut to as few as 8 tokens and each
    option is clipped -- which turns a carefully worded question into noise
    without any error. Every question the tree asks has to fit.
    """

    def collect(self, hand, answers):
        backend = ScriptedBackend(answers)
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(state(hand), raw_g={"hands": {}})
        return backend.log

    def assert_fits(self, question_id, criteria):
        budget = head_budget("Which hand should the player play?", criteria)
        for name, limit in HEAD_BUDGETS.items():
            self.assertLessEqual(
                budget["total"], limit,
                f"{question_id} needs {budget['total']} head tokens, over {name}'s {limit}",
            )

    def test_play_questions_fit_both_checkpoints(self):
        for question_id, criteria in self.collect(
            spades_flush_hand(), {"dig_for_better_hand": False, "send_filler_cards": False}
        ):
            self.assert_fits(question_id, criteria)

    def test_discard_questions_fit_both_checkpoints(self):
        for question_id, criteria in self.collect(
            four_hearts_hand(), {"dig_for_better_hand": True}
        ):
            self.assert_fits(question_id, criteria)

    def test_a_crowded_hand_still_fits(self):
        crowded = [Card("9", "Spades", 0), Card("9", "Hearts", 1), Card("8", "Clubs", 2),
                   Card("8", "Diamonds", 3), Card("7", "Spades", 4), Card("7", "Hearts", 5),
                   Card("6", "Clubs", 6), Card("6", "Diamonds", 7)]
        for question_id, criteria in self.collect(
            crowded, {"dig_for_better_hand": False, "send_filler_cards": False}
        ):
            self.assert_fits(question_id, criteria)

    def test_joker_annotations_still_fit(self):
        """Synergy tags are the longest thing added to an option label."""
        jokers = [
            {"label": "Lusty Joker",
             "value": {"effect": "Played cards with Heart suit give +3 Mult when scored"}},
            {"label": "Scary Face", "value": {"effect": "Played face cards give +30 Chips"}},
            {"label": "Banner", "value": {"effect": "+30 Chips for each remaining discard"}},
        ]
        hand = [Card("K", "Hearts", 0), Card("K", "Clubs", 1), Card("Q", "Hearts", 2),
                Card("Q", "Spades", 3), Card("J", "Hearts", 4), Card("9", "Hearts", 5),
                Card("4", "Clubs", 6), Card("3", "Diamonds", 7)]
        backend = ScriptedBackend({"dig_for_better_hand": False, "send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(state(hand, blind_name="Boss Blind", target_chips=600),
                                 raw_g={"hands": {}, "jokers": {"cards": jokers}})
        for question_id, criteria in backend.log:
            self.assert_fits(question_id, criteria)

    def test_option_count_stays_inside_the_tree_limit(self):
        from sidecar.backends import MAX_OPTIONS_PER_QUESTION

        crowded = [Card("9", "Spades", 0), Card("9", "Hearts", 1), Card("8", "Clubs", 2),
                   Card("8", "Diamonds", 3), Card("7", "Spades", 4), Card("7", "Hearts", 5),
                   Card("6", "Clubs", 6), Card("6", "Diamonds", 7)]
        for question_id, criteria in self.collect(
            crowded, {"dig_for_better_hand": False, "send_filler_cards": False}
        ):
            self.assertLessEqual(len(criteria), MAX_OPTIONS_PER_QUESTION + 1,
                                 f"{question_id} offers {len(criteria)} options")


class TestOtherDecisions(unittest.TestCase):
    def test_joker_reorder_is_proposed_but_not_applied_by_python(self):
        jokers = [
            {"label": "Cavendish", "value": {"effect": "X3 Mult, 1 in 1000 chance to go extinct"}},
            {"label": "Banner", "value": {"effect": "+40 Chips for each remaining discard"}},
            {"label": "Half Joker", "value": {"effect": "+20 Mult if played hand contains 3 or fewer cards"}},
        ]
        keep = BalatroAgent(backend=ScriptedBackend({"joker_order": "order_current"}))
        self.assertIsNone(keep.decide_joker_order(jokers))

        move = BalatroAgent(backend=ScriptedBackend({"joker_order": "order_xmult_last"}))
        self.assertEqual(move.decide_joker_order(jokers), [1, 2, 0])

    def test_planet_is_used_when_the_model_says_so(self):
        agent = BalatroAgent(backend=ScriptedBackend({"consumable_action": "use0_jupiter"}))
        decision = agent.decide_consumable_action([{"label": "Jupiter", "set": "PLANET"}])
        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], 0)

    def test_keeping_consumables_is_respected(self):
        agent = BalatroAgent(backend=ScriptedBackend({"consumable_action": "keep_all"}))
        self.assertIsNone(agent.decide_consumable_action([{"label": "Jupiter", "set": "PLANET"}]))

    def test_pack_skip_is_respected(self):
        agent = BalatroAgent(backend=ScriptedBackend({"pack_choice": "skip_pack"}))
        self.assertIsNone(agent.decide_pack_card([{"label": "Mercury", "set": "PLANET"}]))

    def test_pack_pick_is_respected(self):
        agent = BalatroAgent(backend=ScriptedBackend({"pack_choice": "take1_jupiter"}))
        decision = agent.decide_pack_card([{"label": "Mercury", "set": "PLANET"},
                                           {"label": "Jupiter", "set": "PLANET"}])
        self.assertEqual(decision[0], 1)

    def test_boss_blind_is_never_skipped(self):
        agent = BalatroAgent(backend=ScriptedBackend({"skip_blind": True}))
        raw_g = {"blinds": {
            "small": {"status": "DEFEATED", "name": "Small Blind"},
            "big": {"status": "DEFEATED", "name": "Big Blind"},
            "boss": {"status": "SELECT", "name": "The Hook", "tag_name": "Rare Tag"},
        }}
        action, reasoning = agent.decide_blind_action(state([], target_chips=1000), raw_g)
        self.assertEqual(action, "SELECT")
        self.assertIn("cannot be skipped", reasoning)

    def test_full_joker_slots_ask_which_to_sell(self):
        jokers = [{"label": n, "key": n.lower(), "cost": {"sell": 2}}
                  for n in ("Banner", "Joker", "Half Joker", "Popcorn", "Mystic Summit")]
        backend = ScriptedBackend({
            "shop_category": "cat_joker",
            "sell_joker_for_room": lambda crit: next(k for k in crit if k.startswith("sell2_")),
        })
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        decision = agent.decide_shop_action(
            state([], ante=4, money=10), [{"name": "Cavendish", "cost": 5, "set": "JOKER"}],
            raw_g={"jokers": {"cards": jokers, "count": 5, "max": 5}},
        )
        self.assertEqual(decision[0], "SELL")
        self.assertEqual(decision[1], 2)

    def test_keeping_all_jokers_cancels_the_purchase(self):
        jokers = [{"label": n, "key": n.lower(), "cost": {"sell": 2}}
                  for n in ("Banner", "Joker", "Half Joker", "Popcorn", "Mystic Summit")]
        backend = ScriptedBackend({"shop_category": "cat_joker",
                                   "sell_joker_for_room": "keep_all_jokers"})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        self.assertIsNone(agent.decide_shop_action(
            state([], ante=4, money=10), [{"name": "Cavendish", "cost": 5, "set": "JOKER"}],
            raw_g={"jokers": {"cards": jokers, "count": 5, "max": 5}}))

    def test_eternal_jokers_are_never_offered_for_sale(self):
        jokers = [{"label": "Eternal One", "key": "e1", "modifier": {"eternal": True},
                   "cost": {"sell": 2}}] + [
            {"label": n, "key": n.lower(), "cost": {"sell": 2}}
            for n in ("Joker", "Half Joker", "Popcorn", "Mystic Summit")]
        backend = ScriptedBackend({"shop_category": "cat_joker"})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_shop_action(
            state([], ante=4, money=10), [{"name": "Cavendish", "cost": 5, "set": "JOKER"}],
            raw_g={"jokers": {"cards": jokers, "count": 5, "max": 5}})
        criteria = backend.asked("sell_joker_for_room")[0]
        self.assertNotIn("sell0_eternal_one", criteria)

    def test_empty_lua_modifier_list_does_not_crash(self):
        jokers = [{"label": "Joker", "key": "j", "modifier": [], "cost": {"sell": 1}}] * 5
        backend = ScriptedBackend({"shop_category": "cat_joker",
                                   "sell_joker_for_room": "keep_all_jokers"})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_shop_action(state([], money=10),
                                 [{"name": "Cavendish", "cost": 5, "set": "JOKER"}],
                                 raw_g={"jokers": {"cards": jokers, "count": 5, "max": 5}})


if __name__ == "__main__":
    unittest.main()
