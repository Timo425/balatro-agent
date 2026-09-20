"""Boss blinds change what the cards are worth, so they change the facts.

Before this, a face-down card was reported to the model as the 2 of Spades and a
debuffed card scored its full chips -- so on exactly the blinds that kill runs,
the numbers the model was choosing between were wrong.
"""

import unittest

from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.brief import hand_brief
from sidecar.candidates import enumerate_discard_plans, enumerate_play_cores
from sidecar.client import BalatroBotClient
from sidecar.poker_eval import Card, PokerEvaluator
from tests.support import ScriptedBackend


def raw_card(rank=None, suit=None, hidden=False, debuff=False):
    value = {}
    if rank is not None:
        value["rank"] = rank
    if suit is not None:
        value["suit"] = suit
    return {"value": value, "state": {"hidden": hidden, "debuff": debuff}, "modifier": {}}


class TestParsing(unittest.TestCase):
    def parse(self, cards):
        raw = {"hand": {"cards": cards}, "round": {}, "blinds": {}}
        return BalatroBotClient.parse_state(None, raw).hand

    def test_a_face_down_card_is_not_invented_as_a_two_of_spades(self):
        card = self.parse([raw_card(hidden=True)])[0]
        self.assertTrue(card.face_down)
        self.assertEqual(card.short, "??")

    def test_a_card_with_no_rank_is_treated_as_face_down(self):
        self.assertTrue(self.parse([raw_card()])[0].face_down)

    def test_a_debuffed_card_keeps_its_identity(self):
        card = self.parse([raw_card("K", "Clubs", debuff=True)])[0]
        self.assertFalse(card.face_down)
        self.assertTrue(card.debuffed)
        self.assertIn("debuffed", card.short)

    def test_an_ordinary_card_is_neither(self):
        card = self.parse([raw_card("9", "Hearts")])[0]
        self.assertFalse(card.face_down)
        self.assertFalse(card.debuffed)


class TestDebuffedScoring(unittest.TestCase):
    def test_a_debuffed_card_contributes_no_chips(self):
        cards = [Card("K", "Clubs", 0, debuffed=True), Card("K", "Hearts", 1)]
        self.assertEqual(PokerEvaluator.scoring_chips(cards), 10)
        self.assertEqual(PokerEvaluator.evaluate_5card_score(cards).total_score, (10 + 10) * 2)

    def test_a_debuffed_card_still_makes_the_hand_type(self):
        cards = [Card("K", "Clubs", 0, debuffed=True), Card("K", "Hearts", 1)]
        self.assertEqual(PokerEvaluator.identify_hand(cards)[0], "Pair")

    def test_a_fully_debuffed_hand_scores_the_base_only(self):
        cards = [Card("K", "Clubs", 0, debuffed=True), Card("K", "Diamonds", 1, debuffed=True)]
        self.assertEqual(PokerEvaluator.evaluate_5card_score(cards).total_score, 10 * 2)


class TestFaceDownCards(unittest.TestCase):
    def setUp(self):
        self.hand = [
            Card("K", "Hearts", 0), Card("K", "Spades", 1),
            Card("?", "?", 2, face_down=True), Card("?", "?", 3, face_down=True),
            Card("4", "Clubs", 4), Card("2", "Hearts", 5),
        ]

    def test_no_option_is_built_on_a_face_down_card(self):
        for option in enumerate_play_cores(self.hand):
            self.assertFalse(any(c.face_down for c in option.core), option.key)

    def test_no_discard_plan_counts_a_face_down_card(self):
        for plan in enumerate_discard_plans(self.hand):
            self.assertFalse(any(c.face_down for c in plan.discard), plan.key)

    def test_the_brief_says_how_many_are_face_down(self):
        text = hand_brief(1, "The Mark", 600, 2, 1, self.hand, [],
                          enumerate_play_cores(self.hand))
        self.assertIn("??", text)
        self.assertIn("2 cards are face down", text)

    def test_a_hand_of_only_face_down_cards_offers_nothing(self):
        blind = [Card("?", "?", i, face_down=True) for i in range(3)]
        self.assertEqual(enumerate_play_cores(blind), [])


class TestBossRuleReachesTheModel(unittest.TestCase):
    def test_the_boss_effect_is_stated_in_the_brief(self):
        backend = ScriptedBackend({"which_aim": "play_pair", "send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        hand = [Card("K", "Clubs", 0, debuffed=True), Card("K", "Hearts", 1),
                Card("9", "Spades", 2), Card("4", "Clubs", 3, debuffed=True)]
        state = BalatroGameState(1, "The Club", 600, 0, 2, 1, 5, hand, [])
        raw_g = {"hands": {}, "blinds": {"boss": {"status": "CURRENT", "name": "The Club",
                                                  "effect": "All Club cards are debuffed",
                                                  "score": 600}}}
        facts = agent._hand_facts(state, raw_g)
        self.assertIn("All Club cards are debuffed", facts.play_context)
        self.assertIn("All Club cards are debuffed", facts.discard_context)

    def test_no_rule_line_when_the_blind_has_none(self):
        agent = BalatroAgent(backend=ScriptedBackend({}), tree_mode=True, poll_mode=False)
        state = BalatroGameState(1, "Small Blind", 300, 0, 4, 4, 4,
                                 [Card("K", "Hearts", 0), Card("9", "Spades", 1)], [])
        facts = agent._hand_facts(state, {"hands": {}, "blinds": {
            "small": {"status": "CURRENT", "name": "Small Blind", "effect": "", "score": 300}}})
        self.assertNotIn("This blind's rule", facts.play_context)


if __name__ == "__main__":
    unittest.main()
