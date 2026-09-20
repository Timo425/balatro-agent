"""
Balatro Agent Backends:
Supports Tri-Modal execution:
1. "laya" (Decision Model): Fast local offline ModernBERT-based decision model.
2. "api_llm" (API LLM): Stateless frontier LLM via OpenAI/Anthropic/OpenRouter/Gemini API for automated benchmarking.
3. "agentic" (Agentic AI): Interactive or programmatic agent hook with rich inspection, simulation, and manual/tool override.
"""

import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import httpx

class BaseBackend(ABC):
    """Abstract base class for all Balatro decision backends."""

    @abstractmethod
    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        """Selects one key from criteria based on context and instructions."""
        pass


class LayaBackend(BaseBackend):
    """Laya local offline ModernBERT System 1 decision model."""

    def __init__(self, model_name: str = "convaiinnovations/laya"):
        import laya
        self.model_name = model_name
        self.agent = laya.load(model_name)

    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        questions = {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }
        res = self.agent.predict(context[:2000], questions)
        return res["answers"][question_id]["choice"]


class ApiLlmBackend(BaseBackend):
    """API LLM backend for automated benchmarking (OpenAI, Anthropic, Gemini, OpenRouter, Ollama)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.model = model
        self.timeout = timeout

        # Detect provider from model name or environment
        if api_base:
            self.api_base = api_base.rstrip("/")
        elif "openrouter" in model.lower() or os.environ.get("OPENROUTER_API_KEY"):
            self.api_base = "https://openrouter.ai/api/v1"
        elif "claude" in model.lower() and not os.environ.get("OPENAI_BASE_URL"):
            self.api_base = "https://api.anthropic.com/v1"
        else:
            self.api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        self.client = httpx.Client(timeout=self.timeout)

    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        if not criteria:
            raise ValueError("Criteria cannot be empty")

        options_text = "\n".join(f"- {k}: {desc}" for k, desc in criteria.items())
        valid_keys = list(criteria.keys())

        system_prompt = (
            "You are a grandmaster Balatro AI playing to beat Ante 8 Boss Blind.\n"
            "Evaluate the current game state, joker synergies, probabilities, and card scoring.\n"
            "You MUST select exactly ONE option from the provided candidate keys.\n"
            "Output JSON format only: {\"choice\": \"<selected_key>\", \"reasoning\": \"<brief reason>\"}"
        )

        user_prompt = (
            f"=== CURRENT BALATRO GAME STATE ===\n{context}\n\n"
            f"=== DECISION: {question_id.upper()} ===\n"
            f"Instructions: {instructions}\n\n"
            f"Candidate Actions:\n{options_text}\n\n"
            f"Valid keys: {valid_keys}\n"
            f"Select the single best key to maximize expected chips and survival."
        )

        # Call OpenAI-compatible /v1/chat/completions endpoint
        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"} if "gpt" in self.model else None,
        }

        try:
            resp = self.client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse choice from JSON
            parsed = json.loads(content)
            choice = parsed.get("choice", "")
            if choice in criteria:
                return choice

            # Fallback search for any valid key in content
            for k in valid_keys:
                if f"\"{k}\"" in content or f"'{k}'" in content or re.search(rf"\\b{re.escape(k)}\\b", content):
                    return k
        except Exception as e:
            # If API call fails or no API key, check if first key is available
            pass

        return valid_keys[0]


class AgenticBackend(BaseBackend):
    """Agentic AI backend for interactive play, EV simulation, or external controller hook."""

    def __init__(
        self,
        decision_hook: Optional[Callable[[str, str, str, Dict[str, str]], str]] = None,
        interactive: bool = False,
    ):
        self.decision_hook = decision_hook
        self.interactive = interactive

    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        valid_keys = list(criteria.keys())
        if not valid_keys:
            return ""

        # 1. External hook (e.g. from agent loop or test)
        if self.decision_hook:
            try:
                res = self.decision_hook(context, question_id, instructions, criteria)
                if res in criteria:
                    return res
            except Exception:
                pass

        # 2. Interactive console input
        if self.interactive:
            print(f"\n--- AGENTIC DECISION: {question_id} ---")
            print(f"Instructions: {instructions}")
            for idx, (k, desc) in enumerate(criteria.items(), 1):
                print(f"  [{idx}] {k}: {desc}")
            while True:
                try:
                    inp = input("Enter choice (key or number): ").strip()
                    if inp in criteria:
                        return inp
                    num = int(inp)
                    if 1 <= num <= len(valid_keys):
                        return valid_keys[num - 1]
                except (ValueError, KeyboardInterrupt):
                    break

        # 3. Default to highest priority candidate (first key)
        return valid_keys[0]


def create_backend(
    backend_type: str = "laya",
    model: Optional[str] = None,
    **kwargs,
) -> BaseBackend:
    """Factory function to instantiate the requested backend."""
    b_type = backend_type.lower()
    if b_type in ("laya", "decision_model"):
        return LayaBackend(model_name=model or "convaiinnovations/laya")
    elif b_type in ("api", "api_llm", "llm"):
        return ApiLlmBackend(model=model or "gpt-4o", **kwargs)
    elif b_type in ("agentic", "interactive", "manual"):
        return AgenticBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: '{backend_type}'. Choose 'laya', 'api_llm', or 'agentic'.")
