import unittest
from unittest.mock import MagicMock, patch

from sidecar.backends import (
    AgenticBackend,
    ApiLlmBackend,
    BaseBackend,
    create_backend,
    estimate_tokens,
    head_budget,
)


class TestAgenticBackend(unittest.TestCase):
    def test_hook_answer_is_used(self):
        backend = AgenticBackend(decision_hook=lambda *_: "opt_b")
        self.assertEqual(
            backend.predict_choice("ctx", "q", "choose", {"opt_a": "A", "opt_b": "B"}), "opt_b")
        self.assertEqual(backend.fallbacks, 0)

    def test_no_hook_falls_back_and_says_so(self):
        backend = AgenticBackend()
        self.assertEqual(backend.predict_choice("ctx", "q", "choose", {"a": "1", "b": "2"}), "a")
        self.assertEqual(backend.fallbacks, 1)

    def test_bool_defaults_to_a_two_option_choice(self):
        seen = {}

        class Recording(AgenticBackend):
            def predict_choice(self, context, question_id, instructions, criteria):
                seen.update(criteria)
                return "yes"

        self.assertTrue(Recording().predict_bool("ctx", "q", "is it so?", "yep", "nope"))
        self.assertEqual(seen, {"yes": "yep", "no": "nope"})

    def test_rank_options_default_reports_the_choice(self):
        backend = AgenticBackend(decision_hook=lambda *_: "b")
        scores = backend.rank_options("ctx", "q", lambda k, v: f"pick {v}", {"a": "A", "b": "B"})
        self.assertEqual(max(scores, key=lambda k: scores[k]), "b")


class TestFactory(unittest.TestCase):
    def test_known_backends(self):
        self.assertIsInstance(create_backend("agentic"), AgenticBackend)
        self.assertIsInstance(create_backend("api_llm", model="gpt-4o"), ApiLlmBackend)
        self.assertTrue(create_backend("api_llm_tree", model="gpt-4o").is_tree)
        self.assertFalse(create_backend("api_llm", model="gpt-4o").is_tree)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            create_backend("telepathy")


class TestApiLlmBackend(unittest.TestCase):
    @patch("httpx.Client.post")
    def test_choice_is_parsed_from_json(self, mock_post):
        mock_post.return_value = MagicMock(json=lambda: {
            "choices": [{"message": {"content": '{"choice": "opt_target", "reasoning": "best"}'}}]})
        backend = ApiLlmBackend(model="gpt-4o", api_key="dummy")
        self.assertEqual(
            backend.predict_choice("ctx", "q", "choose", {"opt_other": "o", "opt_target": "t"}),
            "opt_target")
        self.assertEqual(backend.fallbacks, 0)

    @patch("httpx.Client.post", side_effect=RuntimeError("no network"))
    def test_api_failure_is_recorded_as_a_fallback(self, _mock_post):
        backend = ApiLlmBackend(model="gpt-4o", api_key="dummy")
        self.assertEqual(backend.predict_choice("ctx", "q", "choose", {"a": "1", "b": "2"}), "a")
        self.assertEqual(backend.fallbacks, 1)


class TestHeadBudget(unittest.TestCase):
    def test_tokens_grow_with_text(self):
        self.assertLess(estimate_tokens("short"), estimate_tokens("a much longer piece of text"))

    def test_budget_counts_options_and_instructions(self):
        budget = head_budget("Which hand?", {"a": "Pair 9S 9D: ~56 chips, WEAK"})
        self.assertEqual(budget["options"], 1)
        self.assertEqual(budget["total"], budget["option_tokens"] + budget["instruction_tokens"])

    def test_a_wide_question_overruns_the_english_head(self):
        crowded = {f"option_{i}": f"Play some hand number {i} for a middling number of chips"
                   for i in range(12)}
        self.assertGreater(head_budget("Which hand should the player play?", crowded)["total"], 192)


class TestTelemetry(unittest.TestCase):
    def test_telemetry_reports_calls_and_fallbacks(self):
        backend = AgenticBackend()
        backend.predict_choice("ctx", "q", "choose", {"a": "1"})
        info = backend.telemetry()
        self.assertEqual(info["calls"], 1)
        self.assertEqual(info["fallbacks"], 1)

    def test_base_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseBackend()


if __name__ == "__main__":
    unittest.main()
