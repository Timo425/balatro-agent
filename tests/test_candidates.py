import unittest

from sidecar.candidates import (
    annotate_play_options,
    enumerate_discard_plans,
    enumerate_play_cores,
    pick_filler,
)
from sidecar.poker_eval import HAND_TYPE_ORDER, Card


def hand(*specs):
    return [Card(rank, suit, i) for i, (rank, suit) in enumerate(specs)]


class TestPlayEnumeration(unittest.TestCase):
    def test_every_distinct_pair_is_offered(self):
        cards = hand(("Q", "Hearts"), ("9", "Spades"), ("9", "Diamonds"),
                     ("8", "Hearts"), ("8", "Diamonds"), ("7", "Diamonds"),
                     ("6", "Diamonds"), ("3", "Clubs"))
        keys = [o.key for o in enumerate_play_cores(cards)]
        self.assertIn("pair_9", keys)
        self.assertIn("pair_8", keys)
        self.assertIn("two_pair_9_8", keys)

    def test_options_are_in_canonical_hand_order(self):
        cards = hand(("A", "Hearts"), ("K", "Hearts"), ("Q", "Hearts"),
                     ("J", "Hearts"), ("9", "Hearts"), ("9", "Clubs"),
                     ("2", "Spades"), ("3", "Clubs"))
        options = enumerate_play_cores(cards)
        order = [HAND_TYPE_ORDER.index(o.hand_type) for o in options]
        self.assertEqual(order, sorted(order),
                         "enumeration order must be the game's fixed hand ranking")

    def test_flush_and_straight_are_both_found(self):
        cards = hand(("A", "Hearts"), ("K", "Hearts"), ("Q", "Hearts"),
                     ("J", "Hearts"), ("10", "Clubs"), ("2", "Spades"),
                     ("3", "Clubs"), ("4", "Diamonds"))
        types = {o.hand_type for o in enumerate_play_cores(cards)}
        self.assertIn("Straight", types)

    def test_straight_flush_is_found(self):
        cards = hand(("9", "Hearts"), ("10", "Hearts"), ("J", "Hearts"),
                     ("Q", "Hearts"), ("K", "Hearts"), ("2", "Spades"),
                     ("3", "Clubs"), ("4", "Diamonds"))
        types = {o.hand_type for o in enumerate_play_cores(cards)}
        self.assertIn("Straight Flush", types)

    def test_cores_are_deduplicated(self):
        cards = hand(("9", "Hearts"), ("9", "Clubs"), ("4", "Spades"), ("2", "Diamonds"))
        options = enumerate_play_cores(cards)
        signatures = [tuple(sorted(c.id for c in o.core)) for o in options]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_high_card_and_dump_low_card_differ(self):
        cards = hand(("K", "Hearts"), ("3", "Clubs"))
        options = {o.key: o for o in enumerate_play_cores(cards)}
        self.assertEqual(options["high_card"].core[0].rank, "K")
        self.assertEqual(options["dump_low_card"].core[0].rank, "3")

    def test_annotation_does_not_reorder_or_drop(self):
        cards = hand(("Q", "Hearts"), ("9", "Spades"), ("9", "Diamonds"),
                     ("8", "Hearts"), ("8", "Diamonds"))
        options = enumerate_play_cores(cards)
        before = [o.key for o in options]
        after = [o.key for o in annotate_play_options(options, None, [], hands_left=3)]
        self.assertEqual(before, after)

    def test_label_carries_chips_and_strength(self):
        cards = hand(("9", "Spades"), ("9", "Diamonds"), ("2", "Clubs"))
        option = next(o for o in annotate_play_options(
            enumerate_play_cores(cards), None, [], hands_left=4) if o.key == "pair_9")
        label = option.label(needed=300, hands_left=4)
        self.assertIn("~", label)
        self.assertIn("chips", label)

    def test_filler_is_the_cheapest_spare_cards(self):
        cards = hand(("9", "Spades"), ("9", "Diamonds"), ("K", "Clubs"),
                     ("3", "Hearts"), ("2", "Spades"))
        core = [cards[0], cards[1]]
        filler = pick_filler(cards, core)
        self.assertEqual([c.rank for c in filler], ["2", "3", "K"])
        self.assertEqual(len(core) + len(filler), 5)

    def test_empty_hand(self):
        self.assertEqual(enumerate_play_cores([]), [])


class TestDiscardPlans(unittest.TestCase):
    def test_flush_draw_is_offered_with_four_of_a_suit(self):
        cards = hand(("K", "Hearts"), ("10", "Hearts"), ("7", "Hearts"), ("4", "Hearts"),
                     ("2", "Spades"), ("Q", "Clubs"), ("9", "Diamonds"), ("3", "Spades"))
        plans = {p.key: p for p in enumerate_discard_plans(cards)}
        self.assertIn("flush_draw_hearts", plans)
        self.assertTrue(all(c.suit != "Hearts" for c in plans["flush_draw_hearts"].discard))

    def test_keep_pairs_throws_only_singles(self):
        cards = hand(("9", "Spades"), ("9", "Diamonds"), ("8", "Hearts"), ("8", "Clubs"),
                     ("K", "Hearts"), ("4", "Spades"), ("3", "Clubs"), ("2", "Diamonds"))
        plan = next(p for p in enumerate_discard_plans(cards) if p.key == "keep_pairs")
        self.assertTrue(all(c.rank not in ("9", "8") for c in plan.discard))

    def test_joker_suit_plan_appears_when_jokers_reward_it(self):
        cards = hand(("K", "Spades"), ("9", "Spades"), ("7", "Hearts"), ("4", "Clubs"),
                     ("2", "Diamonds"), ("Q", "Clubs"), ("5", "Diamonds"), ("3", "Hearts"))
        plans = {p.key for p in enumerate_discard_plans(cards, favored_suits=["Spades"])}
        self.assertIn("keep_joker_suit_spades", plans)

    def test_throw_lowest_is_always_available(self):
        cards = hand(("K", "Spades"), ("9", "Hearts"), ("7", "Clubs"), ("4", "Diamonds"),
                     ("2", "Spades"), ("Q", "Clubs"), ("5", "Diamonds"), ("3", "Hearts"))
        self.assertIn("throw_lowest", {p.key for p in enumerate_discard_plans(cards)})

    def test_never_discards_more_than_five(self):
        cards = hand(("K", "Spades"), ("9", "Hearts"), ("7", "Clubs"), ("4", "Diamonds"),
                     ("2", "Spades"), ("Q", "Clubs"), ("5", "Diamonds"), ("3", "Hearts"))
        for plan in enumerate_discard_plans(cards):
            self.assertLessEqual(len(plan.discard), 5, plan.key)
            self.assertGreater(len(plan.discard), 0, plan.key)

    def test_plans_are_deduplicated(self):
        cards = hand(("9", "Spades"), ("9", "Hearts"))
        plans = enumerate_discard_plans(cards)
        sigs = [tuple(sorted(c.id for c in p.discard)) for p in plans]
        self.assertEqual(len(sigs), len(set(sigs)))


if __name__ == "__main__":
    unittest.main()
