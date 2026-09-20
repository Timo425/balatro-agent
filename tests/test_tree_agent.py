"""Shape of the hierarchical decision tree: which questions get asked, and when."""

import os
import unittest

from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.backends import MAX_OPTIONS_PER_QUESTION, LayaTreeBackend, create_backend
from sidecar.poker_eval import Card
from tests.support import ScriptedBackend

RUN_LAYA = os.environ.get("BALATRO_TEST_LAYA") == "1"


def state(hand, **overrides):
    base = dict(ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
                hands_left=4, discards_left=3, money=4, hand=hand, jokers=[])
    base.update(overrides)
    return BalatroGameState(**base)


def agent_with(answers, **kwargs):
    backend = ScriptedBackend(answers)
    return BalatroAgent(backend=backend, tree_mode=True, poll_mode=False, **kwargs), backend


class TestTreeShape(unittest.TestCase):
    def test_play_branch_asks_dig_then_which_hand(self):
        hand = [Card("A", "Hearts", 0), Card("K", "Hearts", 1), Card("Q", "Hearts", 2),
                Card("J", "Hearts", 3), Card("9", "Hearts", 4), Card("2", "Spades", 5),
                Card("3", "Clubs", 6), Card("4", "Diamonds", 7)]
        agent, backend = agent_with({"dig_for_better_hand": False,
                                     "which_hand": lambda c: next(k for k in c if k.startswith("flush_")),
                                     "send_filler_cards": False})
        action = agent.decide_hand_action(state(hand), raw_g={"hands": {}})
        self.assertEqual(backend.question_ids[:2], ["dig_for_better_hand", "which_hand"])
        self.assertEqual(len(action.cards), 5)
        self.assertTrue(all(c.suit == "Hearts" for c in action.cards))

    def test_discard_branch_asks_dig_then_which_discard(self):
        hand = [Card("K", "Hearts", 0), Card("10", "Hearts", 1), Card("7", "Hearts", 2),
                Card("4", "Hearts", 3), Card("2", "Spades", 4), Card("Q", "Clubs", 5),
                Card("9", "Diamonds", 6), Card("3", "Spades", 7)]
        agent, backend = agent_with({"dig_for_better_hand": True,
                                     "which_discard": "flush_draw_hearts"})
        action = agent.decide_hand_action(state(hand, target_chips=800), raw_g={"hands": {}})
        self.assertEqual(backend.question_ids, ["dig_for_better_hand", "which_discard"])
        self.assertTrue(all(c.suit != "Hearts" for c in action.cards))

    def test_single_legal_hand_asks_nothing_extra(self):
        agent, backend = agent_with({})
        action = agent.decide_hand_action(
            state([Card("A", "Spades", 0)], discards_left=0), raw_g={"hands": {}})
        self.assertEqual(backend.question_ids, [])
        self.assertEqual(action.action_type, "PLAY")

    def test_crowded_hand_splits_into_type_then_instance(self):
        """Past the option limit the tree goes coarse-to-fine instead of widening."""
        crowded = [Card("9", "Spades", 0), Card("9", "Hearts", 1), Card("8", "Clubs", 2),
                   Card("8", "Diamonds", 3), Card("7", "Spades", 4), Card("7", "Hearts", 5),
                   Card("6", "Clubs", 6), Card("6", "Diamonds", 7)]
        agent, backend = agent_with({"dig_for_better_hand": False,
                                     "which_hand_type": "pair",
                                     "send_filler_cards": False})
        agent.decide_hand_action(state(crowded), raw_g={"hands": {}})
        self.assertIn("which_hand_type", backend.question_ids)
        self.assertIn("which_hand_of_type", backend.question_ids)
        self.assertNotIn("which_hand", backend.question_ids)
        for _, criteria in backend.log:
            self.assertLessEqual(len(criteria), MAX_OPTIONS_PER_QUESTION + 1)

    def test_filler_question_only_for_short_hands(self):
        flush = [Card(r, "Hearts", i) for i, r in enumerate(("A", "K", "Q", "J", "9"))]
        agent, backend = agent_with({"dig_for_better_hand": False})
        agent.decide_hand_action(state(flush, discards_left=0), raw_g={"hands": {}})
        self.assertNotIn("send_filler_cards", backend.question_ids)

    def test_shop_tree_asks_category_then_item(self):
        shop = [{"name": "Joker", "cost": 4, "set": "JOKER"},
                {"name": "Banner", "cost": 5, "set": "JOKER"}]
        agent, backend = agent_with({"shop_category": "cat_joker",
                                     "which_shop_item": "buy1_banner"})
        decision = agent.decide_shop_action(state([], money=10), shop,
                                            raw_g={"jokers": {"cards": []}})
        self.assertEqual(backend.question_ids, ["shop_category", "which_shop_item"])
        self.assertEqual(decision[:2], ("BUY", 1))

    def test_tarot_targets_are_asked_one_at_a_time(self):
        hand = [Card("A", "Spades", 0), Card("K", "Hearts", 1), Card("Q", "Clubs", 2)]
        agent, backend = agent_with({"tarot_target_1": "card0_a", "tarot_target_2": "card1_k"})
        targets = agent.decide_tarot_targets(
            "The Empress", hand=hand, effect="Enhances 2 selected cards to Mult Cards")
        self.assertEqual(targets, [0, 1])
        self.assertEqual(backend.question_ids, ["tarot_target_1", "tarot_target_2"])

    def test_tarot_never_targets_the_same_card_twice(self):
        hand = [Card("A", "Spades", 0), Card("K", "Hearts", 1)]
        agent, _ = agent_with({})
        targets = agent.decide_tarot_targets(
            "The Empress", hand=hand, effect="Enhances 2 selected cards to Mult Cards")
        self.assertEqual(len(set(targets)), len(targets))


class TestPollNode(unittest.TestCase):
    def test_poll_mode_uses_rank_options(self):
        seen = {}

        backend = ScriptedBackend({"dig_for_better_hand": False, "send_filler_cards": False})

        def rank(context, question_id, statement_for, criteria):
            seen[question_id] = {k: statement_for(k, v) for k, v in criteria.items()}
            return {k: float(i) for i, k in enumerate(criteria)}

        backend.rank_options = rank
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=True)
        hand = [Card("9", "Spades", 0), Card("9", "Hearts", 1), Card("K", "Clubs", 2),
                Card("3", "Diamonds", 3)]
        agent.decide_hand_action(state(hand), raw_g={"hands": {}})
        self.assertIn("which_hand", seen)
        for statement in seen["which_hand"].values():
            self.assertIn("should play", statement)


@unittest.skipUnless(RUN_LAYA, "set BALATRO_TEST_LAYA=1 to load the real model")
class TestRealLayaCheckpoint(unittest.TestCase):
    def test_tree_backend_loads_typed_decisions(self):
        backend = create_backend("laya_tree")
        self.assertIsInstance(backend, LayaTreeBackend)
        self.assertTrue(backend.is_tree)
        self.assertEqual(backend.subfolder, "typed-decisions")
        self.assertEqual(backend.max_len, 1024)
        self.assertEqual(backend.head_max_len, 256)

    def test_predictions_come_back_in_range(self):
        backend = create_backend("laya_tree")
        choice = backend.predict_choice("Balatro. Need 300 chips.", "which_hand",
                                        "Which hand should the player play?",
                                        {"pair_9": "Pair 9S 9D: ~56 chips, WEAK",
                                         "high_card": "High Card AD: ~16 chips, TINY"})
        self.assertIn(choice, ("pair_9", "high_card"))
        scores = backend.rank_options(
            "Balatro. Need 300 chips.", "which_hand",
            lambda k, label: f"The player should play {label}",
            {"pair_9": "Pair 9S 9D", "high_card": "High Card AD"})
        self.assertEqual(set(scores), {"pair_9", "high_card"})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))


if __name__ == "__main__":
    unittest.main()
