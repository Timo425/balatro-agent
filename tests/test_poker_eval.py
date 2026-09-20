import unittest

from sidecar.poker_eval import Card, PokerEvaluator


class TestScoringCards(unittest.TestCase):
    """Only the cards that form the hand score in Balatro.

    The four cases below are real scores observed in-game during a losing Ante 1
    run; counting every played card's chips inflated them by 2-3x and made every
    option the model was shown wrong.
    """

    def assert_scores(self, cards, expected_type, expected_total):
        hand_type, scoring = PokerEvaluator.identify_hand(cards)
        self.assertEqual(hand_type, expected_type)
        self.assertEqual(PokerEvaluator.evaluate_5card_score(cards).total_score, expected_total)
        return scoring

    def test_pair_scores_only_the_pair(self):
        cards = [Card("A", "Diamonds"), Card("Q", "Diamonds"), Card("10", "Diamonds"),
                 Card("5", "Spades"), Card("5", "Clubs")]
        scoring = self.assert_scores(cards, "Pair", 40)
        self.assertEqual([c.rank for c in scoring], ["5", "5"])

    def test_two_pair_scores_four_cards(self):
        cards = [Card("Q", "Hearts"), Card("9", "Spades"), Card("9", "Diamonds"),
                 Card("8", "Hearts"), Card("8", "Diamonds")]
        scoring = self.assert_scores(cards, "Two Pair", 108)
        self.assertEqual(len(scoring), 4)
        self.assertNotIn("Q", [c.rank for c in scoring])

    def test_low_pair_with_high_kickers(self):
        cards = [Card("K", "Hearts"), Card("7", "Diamonds"), Card("6", "Diamonds"),
                 Card("4", "Hearts"), Card("4", "Clubs")]
        self.assert_scores(cards, "Pair", 36)

    def test_pair_of_threes(self):
        cards = [Card("K", "Diamonds"), Card("10", "Clubs"), Card("9", "Clubs"),
                 Card("3", "Hearts"), Card("3", "Clubs")]
        self.assert_scores(cards, "Pair", 32)

    def test_flush_scores_all_five(self):
        cards = [Card("A", "Spades"), Card("K", "Spades"), Card("Q", "Spades"),
                 Card("J", "Spades"), Card("9", "Spades")]
        scoring = self.assert_scores(cards, "Flush", (35 + 11 + 10 + 10 + 10 + 9) * 4)
        self.assertEqual(len(scoring), 5)

    def test_three_of_a_kind_scores_three(self):
        cards = [Card("K", "Hearts"), Card("7", "Clubs"), Card("7", "Spades"),
                 Card("7", "Diamonds"), Card("2", "Clubs")]
        scoring = self.assert_scores(cards, "Three of a Kind", (30 + 21) * 3)
        self.assertEqual(len(scoring), 3)

    def test_four_of_a_kind_scores_four(self):
        cards = [Card("K", "Hearts"), Card("7", "Clubs"), Card("7", "Spades"),
                 Card("7", "Diamonds"), Card("7", "Hearts")]
        scoring = self.assert_scores(cards, "Four of a Kind", (60 + 28) * 7)
        self.assertEqual(len(scoring), 4)

    def test_high_card_scores_one(self):
        cards = [Card("K", "Hearts"), Card("9", "Clubs"), Card("7", "Spades"),
                 Card("4", "Diamonds"), Card("2", "Hearts")]
        scoring = self.assert_scores(cards, "High Card", (5 + 10) * 1)
        self.assertEqual(len(scoring), 1)

    def test_wheel_straight(self):
        cards = [Card("A", "Hearts"), Card("2", "Clubs"), Card("3", "Spades"),
                 Card("4", "Diamonds"), Card("5", "Hearts")]
        hand_type, scoring = PokerEvaluator.identify_hand(cards)
        self.assertEqual(hand_type, "Straight")
        self.assertEqual(len(scoring), 5)

    def test_straight_flush(self):
        cards = [Card(r, "Hearts") for r in ("9", "10", "J", "Q", "K")]
        hand_type, _ = PokerEvaluator.identify_hand(cards)
        self.assertEqual(hand_type, "Straight Flush")

    def test_full_house_scores_all_five(self):
        cards = [Card("8", "Hearts"), Card("8", "Clubs"), Card("8", "Spades"),
                 Card("3", "Diamonds"), Card("3", "Hearts")]
        hand_type, scoring = PokerEvaluator.identify_hand(cards)
        self.assertEqual(hand_type, "Full House")
        self.assertEqual(len(scoring), 5)

    def test_two_card_pair_matches_five_card_pair(self):
        """Playing just the pair scores the same as burying it in five cards."""
        pair = [Card("9", "Hearts"), Card("9", "Clubs")]
        padded = pair + [Card("K", "Spades"), Card("4", "Diamonds"), Card("2", "Hearts")]
        self.assertEqual(
            PokerEvaluator.evaluate_5card_score(pair).total_score,
            PokerEvaluator.evaluate_5card_score(padded).total_score,
        )

    def test_live_hand_levels_are_used(self):
        cards = [Card("9", "Hearts"), Card("9", "Clubs")]
        boosted = PokerEvaluator.evaluate_5card_score(cards, level_chips=30, level_mult=4)
        self.assertEqual(boosted.total_score, (30 + 18) * 4)


if __name__ == "__main__":
    unittest.main()
