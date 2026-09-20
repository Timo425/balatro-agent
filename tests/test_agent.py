import unittest
from sidecar.poker_eval import Card
from sidecar.agent import BalatroAgent, BalatroGameState

class TestBalatroAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = BalatroAgent(rules_path="rules/balatro-strategy.md")

    def test_scenario_a_instant_win_flush(self):
        hand = [
            Card("A", "Spades", 1),
            Card("K", "Spades", 2),
            Card("Q", "Spades", 3),
            Card("J", "Spades", 4),
            Card("9", "Spades", 5),
            Card("2", "Hearts", 6),
            Card("3", "Clubs", 7),
            Card("4", "Diamonds", 8)
        ]
        state = BalatroGameState(
            ante=1,
            blind_name="Small Blind",
            target_chips=300,
            current_chips=0,
            hands_left=4,
            discards_left=3,
            money=4,
            hand=hand,
            jokers=[]
        )
        action = self.agent.decide_hand_action(state)
        self.assertEqual(action.action_type, "PLAY")
        self.assertEqual(len(action.cards), 5)
        self.assertTrue(all(c.suit == "Spades" for c in action.cards))

    def test_scenario_b_flush_digging_discard(self):
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
            ante=2,
            blind_name="Big Blind",
            target_chips=800,
            current_chips=0,
            hands_left=4,
            discards_left=3,
            money=10,
            hand=hand,
            jokers=[]
        )
        action = self.agent.decide_hand_action(state)
        self.assertEqual(action.action_type, "DISCARD")
        self.assertTrue(all(c.suit != "Hearts" for c in action.cards))

    def test_rule_joker_order(self):
        jokers = [
            {"label": "Cavendish", "key": "j_cavendish"},
            {"label": "Banner", "key": "j_banner"},
            {"label": "Half Joker", "key": "j_half"}
        ]
        new_order = self.agent.decide_joker_order(jokers)
        self.assertEqual(new_order, [1, 2, 0])

    def test_rule_use_planet(self):
        consumables = [
            {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET"}
        ]
        action = self.agent.decide_consumable_action(consumables)
        self.assertIsNotNone(action)
        idx, reasoning = action
        self.assertEqual(idx, 0)
        self.assertIn("Jupiter", reasoning)

    def test_rule_pack_card_selection_planet(self):
        pack_cards = [
            {"label": "Mercury", "key": "c_mercury", "set": "PLANET"},
            {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET"}
        ]
        decision = self.agent.decide_pack_card(pack_cards)
        self.assertIsNotNone(decision)
        idx, targets, reasoning = decision
        self.assertEqual(idx, 0)  # Mercury is a planet, valid immediately without targets
        self.assertIsNone(targets)

    def test_rule_pack_card_death_with_hand(self):
        pack_cards = [
            {"label": "Death", "key": "c_death", "set": "TAROT"},
            {"label": "Temperance", "key": "c_temperance", "set": "TAROT"}
        ]
        # In this pack, Temperance has priority 2 (no targets), while Death requires 2 targets.
        # Temperance is preferred because it gives cash without risking hand targets
        decision = self.agent.decide_pack_card(pack_cards)
        self.assertIsNotNone(decision)
        idx, targets, reasoning = decision
        self.assertEqual(idx, 1)
        self.assertEqual(targets, None)

if __name__ == "__main__":
    unittest.main()
