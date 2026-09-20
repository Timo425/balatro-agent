import unittest

from sidecar.poker_eval import Card
from sidecar.scoring import (
    ScoreEstimate,
    describe_estimate,
    estimate_play,
    hand_level_values,
    pace_label,
    strength_label,
)


def two_pair():
    return [Card("Q", "Hearts"), Card("9", "Spades"), Card("9", "Diamonds"),
            Card("8", "Hearts"), Card("8", "Diamonds")]


def joker(effect, name="J"):
    return {"label": name, "value": {"effect": effect}}


class TestJokerEstimation(unittest.TestCase):
    def test_no_jokers_matches_base_score(self):
        est = estimate_play(two_pair())
        self.assertEqual(est.total, 108)
        self.assertEqual(est.total, est.base_total)

    def test_flat_mult(self):
        self.assertEqual(estimate_play(two_pair(), None, [joker("+4 Mult")]).total, (54) * 6)

    def test_x_mult(self):
        self.assertEqual(estimate_play(two_pair(), None, [joker("X3 Mult")]).total, 54 * 6)

    def test_x_mult_survives_a_random_downside(self):
        est = estimate_play(two_pair(), None,
                            [joker("X3 Mult, 1 in 1000 chance to go extinct", "Cavendish")])
        self.assertEqual(est.total, 54 * 6)
        self.assertFalse(est.uncertain)

    def test_random_gated_effect_is_not_applied(self):
        est = estimate_play(two_pair(), None, [joker("1 in 4 chance for +10 Mult")])
        self.assertEqual(est.total, 108)
        self.assertTrue(est.uncertain)

    def test_unreadable_effect_is_reported_not_ignored(self):
        est = estimate_play(two_pair(), None, [joker("Retrigger all played cards", "Dusk")])
        self.assertIn("Dusk", est.unmodelled)
        self.assertTrue(est.uncertain)

    def test_per_suit_counts_scoring_cards_only(self):
        # Scoring cards are 9S 9D 8H 8D, so one Diamond nine and one Diamond eight.
        est = estimate_play(two_pair(), None,
                            [joker("Played cards with Diamond suit give +3 Mult when scored")])
        self.assertEqual(est.mult, 2 + 6)

    def test_face_card_joker_ignores_non_scoring_face_cards(self):
        est = estimate_play(two_pair(), None, [joker("Played face cards give +30 Chips")])
        self.assertEqual(est.total, 108, "the Queen kicker does not score")

    def test_hand_type_condition(self):
        pair = [Card("9", "Spades"), Card("9", "Diamonds")]
        self.assertEqual(estimate_play(pair, None, [joker("+30 Chips if played hand is a Flush")]).total,
                         estimate_play(pair).total)

    def test_card_count_condition_uses_cards_played(self):
        short = [Card("9", "Spades"), Card("9", "Diamonds")]
        est = estimate_play(short, None, [joker("+20 Mult if played hand contains 3 or fewer cards")])
        self.assertEqual(est.mult, 22)

    def test_remaining_discards_condition(self):
        est = estimate_play(two_pair(), None, [joker("+30 Chips for each remaining discard")],
                            discards_left=3)
        self.assertEqual(est.chips, 54 + 90)

    def test_debuffed_jokers_are_skipped(self):
        j = joker("X3 Mult")
        j["state"] = {"debuff": True}
        self.assertEqual(estimate_play(two_pair(), None, [j]).total, 108)

    def test_live_hand_levels_are_used(self):
        levels = {"Two Pair": {"chips": 40, "mult": 4, "level": 3}}
        self.assertEqual(estimate_play(two_pair(), levels).total, (40 + 34) * 4)

    def test_hand_level_values_falls_back_to_vanilla(self):
        self.assertEqual(hand_level_values(None, "Flush"), (35, 4, 1))


class TestLabels(unittest.TestCase):
    def test_strength_labels_are_absolute(self):
        self.assertEqual(strength_label(300, 300), "WINS")
        self.assertEqual(strength_label(400, 300), "WINS")
        self.assertEqual(strength_label(200, 300), "STRONG")
        self.assertEqual(strength_label(100, 300), "OK")
        self.assertEqual(strength_label(40, 300), "WEAK")
        self.assertEqual(strength_label(10, 300), "TINY")

    def test_labels_never_compare_options(self):
        for score in (10, 100, 300):
            label = strength_label(score, 300)
            for banned in ("best", "better", "2nd", "worse", "top"):
                self.assertNotIn(banned, label.lower())

    def test_pace_says_when_repeating_still_loses(self):
        self.assertIn("still lose", pace_label(40, 300, 4))
        self.assertIn("hands like it", pace_label(100, 300, 4))
        self.assertEqual(pace_label(300, 300, 4), "wins now")
        self.assertIn("falls short", pace_label(40, 300, 1))

    def test_description_flags_uncertainty(self):
        est = ScoreEstimate("Pair", 20, 2.0, 40, 40, [], ["Mystery Joker"])
        self.assertIn("unread joker text", describe_estimate(est, 300, 4))


if __name__ == "__main__":
    unittest.main()
