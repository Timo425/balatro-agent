"""Routing between a cheap local model and an expensive one.

The point of the hybrid is not a better score -- the slow model still makes
every hard call -- but a cheaper one. So what is under test is where each
question goes, and that nothing silently answers a question it was not trusted
with.
"""

import unittest

from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.backends import HybridBackend, create_backend
from sidecar.poker_eval import Card
from tests.support import ScriptedBackend


class Counting(ScriptedBackend):
    """A scripted backend that remembers what it was asked."""

    def __init__(self, answers=None, confidence=0.9, **kwargs):
        super().__init__(answers, **kwargs)
        self.last_confidence = confidence
        self.seen = []

    def predict_choice(self, context, question_id, instructions, criteria):
        self.seen.append(question_id)
        return super().predict_choice(context, question_id, instructions, criteria)

    def predict_bool(self, context, question_id, statement, true_desc="", false_desc=""):
        self.seen.append(question_id)
        return super().predict_bool(context, question_id, statement, true_desc, false_desc)


def hybrid(fast_answers=None, slow_answers=None, confidence=0.9, **kwargs):
    fast = Counting(fast_answers, confidence=confidence)
    slow = Counting(slow_answers, confidence=0.99)
    kwargs.setdefault("handled", ("which_aim",))
    return HybridBackend(fast=fast, slow=slow, **kwargs), fast, slow


class TestRouting(unittest.TestCase):
    def test_a_trusted_question_goes_to_the_fast_model(self):
        backend, fast, slow = hybrid({"which_aim": "b"})
        self.assertEqual(backend.predict_choice("c", "which_aim", "i", {"a": "A", "b": "B"}), "b")
        self.assertEqual(fast.seen, ["which_aim"])
        self.assertEqual(slow.seen, [])
        self.assertEqual(backend.escalated, 0)

    def test_an_untrusted_question_goes_straight_to_the_slow_model(self):
        backend, fast, slow = hybrid(slow_answers={"shop_category": "cat_save"})
        answer = backend.predict_choice("c", "shop_category", "i",
                                        {"cat_joker": "J", "cat_save": "S"})
        self.assertEqual(answer, "cat_save")
        self.assertEqual(fast.seen, [])
        self.assertEqual(slow.seen, ["shop_category"])
        self.assertEqual(backend.escalated, 1)

    def test_a_hesitant_fast_answer_is_escalated(self):
        backend, fast, slow = hybrid({"which_aim": "b"}, {"which_aim": "a"}, confidence=0.01)
        self.assertEqual(backend.predict_choice("c", "which_aim", "i", {"a": "A", "b": "B"}), "a")
        self.assertEqual(fast.seen, ["which_aim"])
        self.assertEqual(slow.seen, ["which_aim"])

    def test_a_confident_fast_answer_is_kept(self):
        backend, _, slow = hybrid({"which_aim": "b"}, confidence=0.8)
        backend.predict_choice("c", "which_aim", "i", {"a": "A", "b": "B"})
        self.assertEqual(slow.seen, [])

    def test_a_broken_fast_model_escalates_rather_than_failing(self):
        backend, _, slow = hybrid(slow_answers={"which_aim": "a"})
        backend.fast.predict_choice = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        self.assertEqual(backend.predict_choice("c", "which_aim", "i", {"a": "A"}), "a")
        self.assertEqual(slow.seen, ["which_aim"])

    def test_the_trusted_list_can_be_set(self):
        backend, fast, _ = hybrid({"shop_category": "cat_save"}, handled=("shop_category",))
        backend.predict_choice("c", "shop_category", "i", {"cat_save": "S"})
        self.assertEqual(fast.seen, ["shop_category"])

    def test_yes_no_questions_route_the_same_way(self):
        backend, fast, slow = hybrid({"which_aim": True}, handled=("which_aim",))
        self.assertTrue(backend.predict_bool("c", "which_aim", "s", "yes", "no"))
        self.assertEqual(slow.seen, [])
        backend.predict_bool("c", "send_filler_cards", "s", "yes", "no")
        self.assertEqual(slow.seen, ["send_filler_cards"])


class TestTelemetry(unittest.TestCase):
    def test_the_escalation_rate_is_reported(self):
        backend, _, _ = hybrid({"which_aim": "a"})
        backend.predict_choice("c", "which_aim", "i", {"a": "A"})
        backend.predict_choice("c", "shop_category", "i", {"cat_save": "S"})
        info = backend.telemetry()
        self.assertEqual(info["answered_fast"], 1)
        self.assertEqual(info["escalated"], 1)
        self.assertEqual(info["escalation_rate"], 0.5)

    def test_fallbacks_are_the_slow_model_s(self):
        backend, _, slow = hybrid()
        slow.note_fallback()
        self.assertEqual(backend.telemetry()["fallbacks"], 1)

    def test_no_calls_means_no_rate(self):
        backend, _, _ = hybrid()
        self.assertEqual(backend.escalation_rate, 0.0)


class TestWholeTurn(unittest.TestCase):
    def test_a_hand_decision_splits_across_the_two_models(self):
        backend, fast, slow = hybrid({"which_aim": lambda c: next(iter(c))},
                                     {"send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        hand = [Card("Q", "Hearts", 0), Card("9", "Spades", 1), Card("9", "Diamonds", 2),
                Card("8", "Hearts", 3), Card("8", "Diamonds", 4), Card("3", "Clubs", 5)]
        state = BalatroGameState(1, "Small Blind", 300, 0, 4, 4, 4, hand, [])
        agent.decide_hand_action(state, raw_g={"hands": {}})
        self.assertIn("which_aim", fast.seen)
        self.assertIn("send_filler_cards", slow.seen)
        self.assertNotIn("send_filler_cards", fast.seen)


class TestFactory(unittest.TestCase):
    def test_unknown_names_still_rejected(self):
        with self.assertRaises(ValueError):
            create_backend("telepathy")


class TestNothingIsTrustedByDefault(unittest.TestCase):
    """The hand choice was the obvious candidate and it did not survive being
    measured against a plain Python max(), so the default list is empty."""

    def test_default_routes_everything_to_the_slow_model(self):
        fast = Counting({"which_aim": "a"})
        slow = Counting({"which_aim": "b"})
        backend = HybridBackend(fast=fast, slow=slow)
        self.assertEqual(backend.predict_choice("c", "which_aim", "i", {"a": "A", "b": "B"}), "b")
        self.assertEqual(fast.seen, [])
        self.assertEqual(backend.escalation_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
