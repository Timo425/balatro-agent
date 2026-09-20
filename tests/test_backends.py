import unittest
from unittest.mock import MagicMock, patch
from sidecar.backends import BaseBackend, LayaBackend, ApiLlmBackend, AgenticBackend, create_backend
from sidecar.agent import BalatroAgent

class TestBackends(unittest.TestCase):
    def test_agentic_backend_hook(self):
        def my_hook(context, question_id, instructions, criteria):
            return "opt_b"

        backend = AgenticBackend(decision_hook=my_hook)
        choice = backend.predict_choice("test context", "test_q", "choose", {"opt_a": "A", "opt_b": "B"})
        self.assertEqual(choice, "opt_b")

    def test_agentic_backend_default(self):
        backend = AgenticBackend()
        choice = backend.predict_choice("test context", "test_q", "choose", {"opt_1": "1", "opt_2": "2"})
        self.assertEqual(choice, "opt_1")

    def test_create_backend_factory(self):
        b_agentic = create_backend("agentic")
        self.assertIsInstance(b_agentic, AgenticBackend)

        b_api = create_backend("api_llm", model="gpt-4o")
        self.assertIsInstance(b_api, ApiLlmBackend)

    @patch("httpx.Client.post")
    def test_api_llm_backend_parsing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "{\"choice\": \"opt_target\", \"reasoning\": \"best move\"}"}}]
        }
        mock_post.return_value = mock_resp

        backend = ApiLlmBackend(model="gpt-4o", api_key="dummy_key")
        choice = backend.predict_choice(
            "context",
            "shop_action",
            "choose",
            {"opt_other": "other", "opt_target": "target"}
        )
        self.assertEqual(choice, "opt_target")

    def test_balatro_agent_with_agentic_backend(self):
        backend = AgenticBackend(decision_hook=lambda *args: "play_flush")
        agent = BalatroAgent(backend=backend)
        self.assertEqual(agent.backend, backend)

if __name__ == "__main__":
    unittest.main()
