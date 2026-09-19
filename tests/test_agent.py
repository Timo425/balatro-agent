import unittest
from sidecar.poker_eval import Card
from sidecar.agent import BalatroAgent, BalatroGameState

class TestBalatroAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = BalatroAgent(rules_path="rules/balatro-strategy.md")

    def test_scenario_a_instant_win_flush(self):
        """Scenario A: Hand contains an immediate winning Flush against Ante 1 Small Blind (300)."""
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
        print("\n--- Scenario A (Winning Flush) ---")
        print("Action:", action.action_type)
        print("Reasoning:", action.reasoning)
        print("Cards:", action.cards)

        self.assertEqual(action.action_type, "PLAY")
        self.assertEqual(len(action.cards), 5)
        self.assertTrue(all(c.suit == "Spades" for c in action.cards))

    def test_scenario_b_flush_digging_discard(self):
        """Scenario B: Weak hand (High Card only) against 800 chip blind, holding 4 Hearts with 3 discards remaining."""
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
        print("\n--- Scenario B (Flush Digging) ---")
        print("Action:", action.action_type)
        print("Reasoning:", action.reasoning)
        print("Cards:", action.cards)

        self.assertEqual(action.action_type, "DISCARD")
        # Discarded cards must NOT be Hearts
        self.assertTrue(all(c.suit != "Hearts" for c in action.cards))

    def test_scenario_c_shop_joker_purchase(self):
        """Scenario C: Shop in Ante 1 with $12, holding 0 Jokers."""
        state = BalatroGameState(
            ante=1,
            blind_name="Small Blind",
            target_chips=300,
            current_chips=350,
            hands_left=3,
            discards_left=2,
            money=12,
            hand=[],
            jokers=[]
        )
        shop_jokers = ["Gros Michel (+15 Mult)", "Banner (+40 Chips per discard)"]
        decision = self.agent.decide_shop_purchase(state, shop_jokers)
        print("\n--- Scenario C (Shop Decision) ---")
        print("Shop Decision:", decision)
        self.assertIn(decision, ["buy_flat_mult", "buy_scaling"])

if __name__ == "__main__":
    unittest.main()
