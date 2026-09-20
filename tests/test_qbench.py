"""The question benchmark: items must have answers that are actually right."""

import unittest

from sidecar.qbench import QItem, build_items, chance_level, evaluate, strip_labels
from sidecar.stress import ProbeBackend


class TestStripLabels(unittest.TestCase):
    def test_verdict_and_numbers_are_removed(self):
        bare = strip_labels("WINS - Flush AS KS QS JS 9S: ~340 chips, wins now")
        self.assertEqual(bare, "Flush AS KS QS JS 9S")

    def test_bare_text_is_left_alone(self):
        self.assertEqual(strip_labels("Pair 9S 9D"), "Pair 9S 9D")

    def test_every_verdict_word_is_stripped(self):
        for verdict in ("WINS", "STRONG", "OK", "WEAK", "TINY"):
            bare = strip_labels(f"{verdict} - Pair 9S 9D: ~56 chips, 3 more like it still lose")
            self.assertEqual(bare, "Pair 9S 9D")


class TestItems(unittest.TestCase):
    def setUp(self):
        self.items = build_items(range(12))

    def test_both_variants_are_produced_for_every_question(self):
        kinds = {i.kind for i in self.items}
        for kind in kinds:
            variants = {i.variant for i in self.items if i.kind == kind}
            self.assertEqual(variants, {"labelled", "bare"}, kind)

    def test_the_answer_is_always_one_of_the_options(self):
        for item in self.items:
            if item.qtype == "noul":
                self.assertIn(item.answer, (True, False))
            else:
                self.assertIn(item.answer, item.criteria, item.kind)

    def test_bare_items_carry_no_numbers(self):
        for item in self.items:
            if item.variant != "bare":
                continue
            for text in item.criteria.values():
                self.assertNotIn("~", text, item.kind)

    def test_yes_and_no_answers_are_balanced(self):
        for kind in ("beats_the_blind", "dig_or_play"):
            answers = [i.answer for i in self.items if i.kind == kind]
            self.assertEqual(answers.count(True), answers.count(False), kind)

    def test_chance_level_matches_the_option_count(self):
        self.assertAlmostEqual(chance_level(self.items, "bigger_of_two", "labelled"), 0.5)


class TestEvaluation(unittest.TestCase):
    def test_a_random_probe_lands_near_chance(self):
        items = build_items(range(30))
        results = evaluate(ProbeBackend(policy="random", seed=1), items)
        for (kind, variant), res in results.items():
            chance = chance_level(items, kind, variant)
            self.assertLess(abs(res.accuracy - chance), 0.35, f"{kind}/{variant}")

    def test_results_count_every_item(self):
        items = build_items(range(8))
        results = evaluate(ProbeBackend(policy="first"), items)
        self.assertEqual(sum(r.items for r in results.values()), len(items))


if __name__ == "__main__":
    unittest.main()
