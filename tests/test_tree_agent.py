import unittest
from unittest.mock import MagicMock
from sidecar.backends import create_backend, LayaTreeBackend, AgenticBackend
from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.poker_eval import Card

class TestTreeAgent(unittest.TestCase):
    def test_laya_tree_backend_creation(self):
        backend = create_backend("laya_tree")
        self.assertIsInstance(backend, LayaTreeBackend)
        self.assertTrue(getattr(backend, "is_tree", False))
        self.assertEqual(backend.agent.cfg.get("max_len"), 1024)
        self.assertEqual(backend.agent.cfg.get("head_max_len"), 256)
        self.assertEqual(backend.max_chars, 4096)

    def test_agent_auto_enables_tree_mode(self):
        backend = AgenticBackend()
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)
        self.assertTrue(agent.tree_mode)

    def test_hand_action_tree_intent_play(self):
        queries = []
        def mock_hook(context, question_id, instructions, criteria):
            queries.append(question_id)
            if question_id == "macro_intent":
                return "intent_play"
            elif question_id == "play_hand":
                # Find flush candidate
                for k in criteria:
                    if "flush" in k:
                        return k
                return next(iter(criteria))
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        hand = [
            Card("A", "Hearts", 1),
            Card("K", "Hearts", 2),
            Card("Q", "Hearts", 3),
            Card("J", "Hearts", 4),
            Card("9", "Hearts", 5),
            Card("2", "Spades", 6),
            Card("3", "Clubs", 7),
            Card("4", "Diamonds", 8)
        ]
        state = BalatroGameState(
            ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
            hands_left=4, discards_left=3, money=4, hand=hand, jokers=[]
        )
        action = agent.decide_hand_action(state)
        self.assertEqual(action.action_type, "PLAY")
        self.assertEqual(len(action.cards), 5)
        self.assertTrue(all(c.suit == "Hearts" for c in action.cards))
        self.assertIn("macro_intent", queries)
        self.assertIn("play_hand", queries)

    def test_hand_action_tree_intent_discard(self):
        queries = []
        def mock_hook(context, question_id, instructions, criteria):
            queries.append(question_id)
            if question_id == "macro_intent":
                return "intent_discard"
            elif question_id == "discard_hand":
                return "discard_offsuit" if "discard_offsuit" in criteria else next(iter(criteria))
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        hand = [
            Card("K", "Hearts", 1),
            Card("10", "Hearts", 2),
            Card("7", "Hearts", 3),
            Card("4", "Hearts", 4),
            Card("2", "Spades", 5),
            Card("Q", "Clubs", 6),
            Card("9", "Diamonds", 7),
            Card("3", "Spades", 8)
        ]
        state = BalatroGameState(
            ante=2, blind_name="Big Blind", target_chips=800, current_chips=0,
            hands_left=4, discards_left=3, money=10, hand=hand, jokers=[]
        )
        action = agent.decide_hand_action(state)
        self.assertEqual(action.action_type, "DISCARD")
        self.assertTrue(all(c.suit != "Hearts" for c in action.cards))
        self.assertIn("macro_intent", queries)
        self.assertIn("discard_hand", queries)

    def test_hand_action_tree_bypasses_intent_when_no_discards_left(self):
        queries = []
        def mock_hook(context, question_id, instructions, criteria):
            queries.append(question_id)
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        hand = [Card("A", "Spades", 1), Card("K", "Hearts", 2)]
        state = BalatroGameState(
            ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
            hands_left=1, discards_left=0, money=4, hand=hand, jokers=[]
        )
        action = agent.decide_hand_action(state)
        self.assertEqual(action.action_type, "PLAY")
        # Macro intent must NOT have been queried because discards_left == 0
        self.assertNotIn("macro_intent", queries)

    def test_shop_action_tree_buy_joker(self):
        queries = []
        def mock_hook(context, question_id, instructions, criteria):
            queries.append(question_id)
            if question_id == "shop_category":
                return "cat_joker"
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        state = BalatroGameState(
            ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
            hands_left=4, discards_left=3, money=10, hand=[], jokers=[]
        )
        shop_items = [{"name": "Joker", "cost": 4, "set": "JOKER"}]
        decision = agent.decide_shop_action(state, shop_items, max_jokers=5)
        self.assertIsNotNone(decision)
        act_type, idx, desc = decision
        self.assertEqual(act_type, "BUY")
        self.assertEqual(idx, 0)
        self.assertIn("shop_category", queries)

    def test_shop_action_tree_save_money(self):
        def mock_hook(context, question_id, instructions, criteria):
            return "cat_save"

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        state = BalatroGameState(
            ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
            hands_left=4, discards_left=3, money=10, hand=[], jokers=[]
        )
        shop_items = [{"name": "Joker", "cost": 4, "set": "JOKER"}]
        decision = agent.decide_shop_action(state, shop_items, max_jokers=5)
        self.assertIsNone(decision)

    def test_shop_action_tree_slot_management_sell_when_full(self):
        def mock_hook(context, question_id, instructions, criteria):
            if question_id == "shop_category":
                return "cat_joker"
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        state = BalatroGameState(
            ante=4, blind_name="Small Blind", target_chips=5000, current_chips=0,
            hands_left=4, discards_left=3, money=10, hand=[], jokers=["Joker", "Half Joker", "Popcorn", "Mystic Summit", "Banner"]
        )
        raw_g = {
            "jokers": {
                "count": 5, "max": 5,
                "cards": [
                    {"label": "Banner", "key": "j_banner", "cost": {"sell": 2}},
                    {"label": "Joker", "key": "j_joker", "cost": {"sell": 1}},
                    {"label": "Half Joker", "key": "j_half", "cost": {"sell": 2}},
                    {"label": "Popcorn", "key": "j_popcorn", "cost": {"sell": 2}},
                    {"label": "Mystic Summit", "key": "j_mystic", "cost": {"sell": 2}}
                ]
            }
        }
        shop_items = [{"name": "Cavendish", "cost": 5, "set": "JOKER"}]
        decision = agent.decide_shop_action(state, shop_items, max_jokers=5, raw_g=raw_g)
        self.assertIsNotNone(decision)
        act_type, idx, desc = decision
        self.assertEqual(act_type, "SELL")
        self.assertEqual(idx, 0)
        self.assertIn("Sell", desc)


    def test_tarot_two_targets_tree_selection(self):
        queries = []
        def mock_hook(context, question_id, instructions, criteria):
            queries.append(question_id)
            if question_id == "tarot_target":
                return "card_0"
            elif question_id == "tarot_target_2":
                return "card_1"
            return next(iter(criteria))

        backend = AgenticBackend(decision_hook=mock_hook)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend)

        hand = [Card("A", "Spades", 0), Card("K", "Hearts", 1), Card("Q", "Clubs", 2)]
        targets = agent.decide_tarot_targets("The Empress", hand=hand, effect="Enhances 2 selected cards to Mult Cards")
        self.assertEqual(targets, [0, 1])
        self.assertIn("tarot_target", queries)
        self.assertIn("tarot_target_2", queries)

if __name__ == "__main__":
    unittest.main()
