import unittest

from sidecar.agent import BalatroAgent
from sidecar.poker_eval import Card
from sidecar.sim import JOKER_LIBRARY, BalatroSim, blind_target
from sidecar.stress import ProbeBackend, run_stress


class TestBlindTargets(unittest.TestCase):
    def test_ante_one(self):
        self.assertEqual(blind_target(1, "Small Blind"), 300)
        self.assertEqual(blind_target(1, "Big Blind"), 450)
        self.assertEqual(blind_target(1, "Boss Blind"), 600)

    def test_ante_two(self):
        self.assertEqual(blind_target(2, "Small Blind"), 800)
        self.assertEqual(blind_target(2, "Boss Blind"), 1600)


class TestSimScoring(unittest.TestCase):
    def test_ground_truth_matches_the_game_for_a_pair(self):
        sim = BalatroSim(seed=1)
        cards = [Card("A", "Diamonds"), Card("Q", "Diamonds"), Card("10", "Diamonds"),
                 Card("5", "Spades"), Card("5", "Clubs")]
        self.assertEqual(sim.score(cards, hands_left=4, discards_left=4), 40)

    def test_joker_rules_are_independent_of_the_text_estimator(self):
        sim = BalatroSim(seed=1, jokers=[JOKER_LIBRARY["banner"]])
        cards = [Card("9", "Spades"), Card("9", "Diamonds")]
        # (10 base + 18 card chips + 30*2 discards) * 2 mult
        self.assertEqual(sim.score(cards, hands_left=4, discards_left=2), (10 + 18 + 60) * 2)

    def test_deck_is_a_full_52(self):
        deck = BalatroSim(seed=3).fresh_deck()
        self.assertEqual(len(deck), 52)
        self.assertEqual(len({(c.rank, c.suit) for c in deck}), 52)

    def test_same_seed_deals_the_same_deck(self):
        self.assertEqual([c.short for c in BalatroSim(seed=7).fresh_deck()],
                         [c.short for c in BalatroSim(seed=7).fresh_deck()])


class TestSimExecutesWhatItIsGiven(unittest.TestCase):
    def test_a_round_terminates_and_reports_honestly(self):
        backend = ProbeBackend(policy="first")
        backend.is_tree = True
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        result = BalatroSim(seed=11).play_round(agent, ante=1, blind="Small Blind")
        self.assertLessEqual(result.hands_used, 4)
        self.assertLessEqual(result.discards_used, 4)
        self.assertEqual(result.won, result.chips >= result.target)

    def test_run_stops_at_the_first_lost_blind(self):
        backend = ProbeBackend(policy="random", seed=2)
        backend.is_tree = True
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        run = BalatroSim(seed=5).play_run(agent, max_ante=2)
        self.assertTrue(all(r.won for r in run.rounds[:-1]))


class TestProbesRankAsExpected(unittest.TestCase):
    """The label probe reads only what the model reads, so it measures the tree."""

    def test_label_probe_beats_random_and_first(self):
        seeds = range(12)
        label = run_stress("label", "tree", seeds, max_ante=3)
        random_probe = run_stress("random", "tree", seeds, max_ante=3)
        first = run_stress("first", "tree", seeds, max_ante=3)
        self.assertGreater(sum(label.rounds_won), sum(random_probe.rounds_won))
        self.assertGreater(sum(label.rounds_won), sum(first.rounds_won))

    def test_tree_gives_the_label_reader_more_than_flat(self):
        seeds = range(20)
        tree = run_stress("label", "tree", seeds, max_ante=3)
        flat = run_stress("label", "flat", seeds, max_ante=3)
        self.assertGreater(sum(tree.rounds_won), sum(flat.rounds_won))

    def test_no_fallbacks_when_the_backend_answers(self):
        report = run_stress("label", "tree", range(5), max_ante=2)
        self.assertEqual(report.fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
