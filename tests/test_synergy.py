import unittest
from sidecar.synergy import JokerSynergyAnalyzer, SynergyProfile
from sidecar.poker_eval import Card


class TestJokerSynergyAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = JokerSynergyAnalyzer()

    # --- analyze_joker tests ---

    def test_suit_detection_lusty_joker(self):
        joker = {"label": "Lusty Joker", "value": {"effect": "+3 Mult if played hand contains a Heart card"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertEqual(syn.suits, ["Hearts"])
        self.assertFalse(syn.face_cards)

    def test_hand_type_detection_flush(self):
        joker = {"label": "Splash", "value": {"effect": "+5 Mult if played hand is a Flush"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertEqual(syn.hand_types, ["Flush"])

    def test_card_count_detection(self):
        joker = {"label": "Half Joker", "value": {"effect": "+20 Mult if played hand contains 3 or fewer cards"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertEqual(syn.card_counts, [3])

    def test_face_card_detection(self):
        joker = {"label": "Scary Face", "value": {"effect": "Played face cards give +30 Chips"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertTrue(syn.face_cards)

    def test_resource_detection_discard(self):
        joker = {"label": "Banner", "value": {"effect": "+40 Chips for each remaining discard"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertIn("discard", syn.resources)

    def test_no_synergy_for_generic_joker(self):
        joker = {"label": "Cavendish", "value": {"effect": "X3 Mult, 1 in 1000 chance to go extinct"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertEqual(syn.suits, [])
        self.assertEqual(syn.hand_types, [])
        self.assertFalse(syn.face_cards)

    def test_multiple_suits(self):
        joker = {"label": "Smeared Joker", "value": {"effect": "Hearts and Diamonds count as the same suit"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertIn("Hearts", syn.suits)
        self.assertIn("Diamonds", syn.suits)

    def test_tags_property(self):
        joker = {"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}
        syn = self.analyzer.analyze_joker(joker)
        self.assertIn("Hearts", syn.tags)

    # --- analyze (aggregate profile) tests ---

    def test_empty_joker_list(self):
        profile = self.analyzer.analyze([])
        self.assertFalse(profile.has_synergies)
        self.assertEqual(profile.summary, "no specific synergy")

    def test_profile_aggregates_suits(self):
        jokers = [
            {"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}},
            {"label": "Greedy Joker", "value": {"effect": "+3 Mult per Diamond card"}},
        ]
        profile = self.analyzer.analyze(jokers)
        self.assertIn("Hearts", profile.favored_suits)
        self.assertIn("Diamonds", profile.favored_suits)
        self.assertTrue(profile.has_synergies)

    def test_profile_deduplicates(self):
        jokers = [
            {"label": "J1", "value": {"effect": "+3 Mult per Heart card"}},
            {"label": "J2", "value": {"effect": "+5 Chips per Heart card"}},
        ]
        profile = self.analyzer.analyze(jokers)
        self.assertEqual(profile.favored_suits.count("Hearts"), 1)

    def test_debuffed_joker_excluded(self):
        jokers = [
            {"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart"}, "state": {"debuff": True}},
        ]
        profile = self.analyzer.analyze(jokers)
        self.assertFalse(profile.has_synergies)

    def test_profile_summary(self):
        jokers = [
            {"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}},
            {"label": "Splash", "value": {"effect": "+5 Mult if played hand is a Flush"}},
        ]
        profile = self.analyzer.analyze(jokers)
        self.assertIn("Hearts", profile.summary)
        self.assertIn("Flush", profile.summary)

    # --- annotate_hand tests ---

    def test_annotate_hand_suit_match(self):
        jokers = [{"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}]
        profile = self.analyzer.analyze(jokers)
        cards = [Card("A", "Hearts", 1), Card("K", "Hearts", 2), Card("Q", "Hearts", 3)]
        annotation = self.analyzer.annotate_hand(profile, cards, "Three of a Kind")
        self.assertIn("🃏", annotation)
        self.assertIn("Hearts", annotation)
        self.assertIn("Lusty Joker", annotation)

    def test_annotate_hand_type_match(self):
        jokers = [{"label": "Splash", "value": {"effect": "+5 Mult if played hand is a Flush"}}]
        profile = self.analyzer.analyze(jokers)
        cards = [Card("A", "Hearts", i) for i in range(5)]
        annotation = self.analyzer.annotate_hand(profile, cards, "Flush")
        self.assertIn("Flush", annotation)
        self.assertIn("Splash", annotation)

    def test_annotate_hand_no_match(self):
        jokers = [{"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}]
        profile = self.analyzer.analyze(jokers)
        cards = [Card("A", "Spades", 1), Card("K", "Spades", 2)]
        annotation = self.analyzer.annotate_hand(profile, cards, "Pair")
        self.assertEqual(annotation, "")

    def test_annotate_hand_face_card_match(self):
        jokers = [{"label": "Scary Face", "value": {"effect": "Played face cards give +30 Chips"}}]
        profile = self.analyzer.analyze(jokers)
        cards = [Card("K", "Hearts", 1), Card("Q", "Spades", 2)]
        annotation = self.analyzer.annotate_hand(profile, cards, "Pair")
        self.assertIn("Face", annotation)
        self.assertIn("Scary Face", annotation)

    def test_annotate_hand_card_count_match(self):
        jokers = [{"label": "Half Joker", "value": {"effect": "+20 Mult if played hand contains 3 or fewer cards"}}]
        profile = self.analyzer.analyze(jokers)
        cards = [Card("A", "Hearts", 1), Card("K", "Hearts", 2)]  # 2 cards <= 3
        annotation = self.analyzer.annotate_hand(profile, cards, "Pair")
        self.assertIn("Half Joker", annotation)

    def test_annotate_hand_empty_profile(self):
        profile = SynergyProfile()
        cards = [Card("A", "Hearts", 1)]
        annotation = self.analyzer.annotate_hand(profile, cards, "High Card")
        self.assertEqual(annotation, "")

    # --- annotate_shop_item tests ---

    def test_annotate_shop_item_reinforces(self):
        jokers = [{"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}]
        profile = self.analyzer.analyze(jokers)
        annotation = self.analyzer.annotate_shop_item(
            profile, "+5 Chips per Heart card scored", "Greedy Hearts"
        )
        self.assertIn("reinforces", annotation)
        self.assertIn("Hearts", annotation)

    def test_annotate_shop_item_new_synergy(self):
        jokers = [{"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}]
        profile = self.analyzer.analyze(jokers)
        annotation = self.analyzer.annotate_shop_item(
            profile, "+4 Mult if played hand is a Flush", "Splash"
        )
        self.assertIn("adds", annotation)
        self.assertIn("Flush", annotation)

    def test_annotate_shop_item_no_effect(self):
        profile = SynergyProfile()
        annotation = self.analyzer.annotate_shop_item(profile, "", "Random Item")
        self.assertEqual(annotation, "")

    # --- joker_contribution tests ---

    def test_joker_contribution_with_synergy(self):
        jokers = [{"label": "Lusty Joker", "value": {"effect": "+3 Mult per Heart card"}}]
        profile = self.analyzer.analyze(jokers)
        contribution = self.analyzer.joker_contribution(profile, "Lusty Joker", "+3 Mult per Heart card")
        self.assertIn("Hearts", contribution)
        self.assertIn("synergy", contribution)

    def test_joker_contribution_no_synergy(self):
        jokers = [{"label": "Cavendish", "value": {"effect": "X3 Mult"}}]
        profile = self.analyzer.analyze(jokers)
        contribution = self.analyzer.joker_contribution(profile, "Cavendish", "X3 Mult")
        self.assertEqual(contribution, "no detected synergies")


if __name__ == "__main__":
    unittest.main()
