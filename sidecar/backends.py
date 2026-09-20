"""Balatro decision backends.

Every backend answers the same two primitives, and every in-game action is the
direct result of one of these answers:

- `predict_choice` -> picks one key out of a small set of labelled options
- `predict_bool`   -> answers a yes/no question

Four implementations:
1. "laya"       -- local Laya decision model, flat (one question per decision)
2. "laya_tree"  -- local Laya decision model, hierarchical (small questions in sequence)
3. "api_llm"    -- stateless frontier LLM over an OpenAI-compatible API
4. "agentic"    -- an external agent or a human answering interactively

Python supplies options and facts. The backend supplies the decision. If a
backend fails, the fallback is recorded in `fallbacks` and surfaced in telemetry:
a run with fallbacks is Python's play, not the model's, and must not be reported
as a model result. See CLAUDE.md.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

# Laya shares one token budget between a question's options and its instructions:
# when the options overrun it, the instructions are truncated to as few as 8
# tokens. Keeping questions under these limits is what stops that happening.
MAX_OPTIONS_PER_QUESTION = 6
MAX_OPTION_TOKENS = 30


def estimate_tokens(text: str) -> int:
    """Rough token count for budget checks (~3.6 characters per token)."""
    return max(1, int(len(str(text)) / 3.6) + 1)


def head_budget(instructions: str, criteria: Dict[str, str]) -> Dict[str, int]:
    """How many head tokens a question would use, the way Laya lays it out.

    Laya builds `[CLS] instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state`,
    and the options are charged first. When they overrun `head_max_len` the
    instructions are cut down to as little as 8 tokens and each option is clipped
    to a few characters, which is how a carefully worded prompt silently becomes
    noise. Use this to keep questions inside the budget.
    """
    option_tokens = sum(estimate_tokens(f"{k}: {v}") + 1 for k, v in criteria.items())
    return {
        "options": len(criteria),
        "option_tokens": option_tokens,
        "instruction_tokens": estimate_tokens(instructions),
        "total": option_tokens + estimate_tokens(instructions),
    }


class BaseBackend(ABC):
    """Abstract base class for all Balatro decision backends."""

    #: Backends that can read a long markdown gamestate rather than a compact brief.
    wants_full_context: bool = False
    is_tree: bool = False
    #: Token budget shared by a question's options and instructions. Only the
    #: local decision model has a real limit here; LLMs and humans do not.
    head_max_len: int = 4096

    def __init__(self) -> None:
        self.calls = 0
        self.fallbacks = 0
        self.last_confidence: Optional[float] = None

    @abstractmethod
    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        """Selects one key from criteria based on context and instructions."""

    def predict_bool(
        self,
        context: str,
        question_id: str,
        statement: str,
        true_desc: str = "",
        false_desc: str = "",
    ) -> bool:
        """Answers a yes/no question. Defaults to a two-option choice."""
        criteria = {
            "yes": true_desc or "yes, the statement holds",
            "no": false_desc or "no, the statement does not hold",
        }
        return self.predict_choice(context, question_id, statement, criteria) == "yes"

    def rank_options(
        self,
        context: str,
        question_id: str,
        statement_for: Callable[[str, str], str],
        criteria: Dict[str, str],
    ) -> Dict[str, float]:
        """Scores every option independently and returns the model's own scores.

        The default implementation just asks the ordinary choice question and
        reports the winner, so backends that have no per-option scoring still work.
        """
        chosen = self.predict_choice(
            context, question_id, statement_for("", ""), criteria
        )
        return {key: (1.0 if key == chosen else 0.0) for key in criteria}

    def note_fallback(self) -> None:
        self.fallbacks += 1

    def telemetry(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "fallbacks": self.fallbacks,
            "last_confidence": self.last_confidence,
        }


class LayaBackend(BaseBackend):
    """Laya: a local, non-autoregressive System 1 decision model.

    Laya never generates text; it scores each option at its own marker token. It
    cannot do arithmetic, so every number it needs must already be in the option
    text. It also has a hard option-token budget -- see `MAX_OPTIONS_PER_QUESTION`.
    """

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        subfolder: Optional[str] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        import laya

        self.model_name = model_name
        self.subfolder = subfolder
        self.agent = laya.load(model_name, device=device, subfolder=subfolder)
        # Only override the checkpoint's own budgets when explicitly asked: the
        # English root was trained at 512/192 and the typed-decisions checkpoint at
        # 1024/256, and feeding one the other's budget wastes both.
        if max_len is not None:
            self.agent.cfg["max_len"] = max_len
        if head_max_len is not None:
            self.agent.cfg["head_max_len"] = head_max_len
        self.max_len = self.agent.cfg.get("max_len", 512)
        self.head_max_len = self.agent.cfg.get("head_max_len", 192)
        # Roughly 4 characters per token, minus the head the options occupy.
        self.max_chars = max(400, (self.max_len - self.head_max_len) * 4)

    def _ask(self, context: str, question_id: str, qdef: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        res = self.agent.predict(context[: self.max_chars], {question_id: qdef})
        answer = res["answers"][question_id]
        self.last_confidence = answer.get("confidence")
        return answer

    def predict_choice(
        self,
        context: str,
        question_id: str,
        instructions: str,
        criteria: Dict[str, str],
    ) -> str:
        answer = self._ask(
            context,
            question_id,
            {"type": "choice", "instructions": instructions, "criteria": criteria},
        )
        return answer["choice"]

    def predict_bool(
        self,
        context: str,
        question_id: str,
        statement: str,
        true_desc: str = "",
        false_desc: str = "",
    ) -> bool:
        qdef: Dict[str, Any] = {"type": "noul", "instructions": statement}
        if true_desc or false_desc:
            qdef["criteria"] = {"true": true_desc, "false": false_desc}
        answer = self._ask(context, question_id, qdef)
        return float(answer["noul"]) >= 0.5


    def rank_options(
        self,
        context: str,
        question_id: str,
        statement_for: Callable[[str, str], str],
        criteria: Dict[str, str],
    ) -> Dict[str, float]:
        """Poll: one yes/no question per option, all in a single forward pass.

        `noul` is Laya's strongest primitive (0.857 against 0.733 for `choice` on
        its own benchmark) and a two-option question never runs out of head budget,
        however many candidates there are. Every number below comes from the model;
        picking the highest is reading its answer, not overruling it.
        """
        if not criteria:
            return {}
        questions = {
            key: {"type": "noul", "instructions": statement_for(key, label)}
            for key, label in criteria.items()
        }
        self.calls += 1
        result = self.agent.predict(context[: self.max_chars], questions)
        scores = {key: float(result["answers"][key]["noul"]) for key in criteria}
        confidences = [result["answers"][key].get("confidence", 0.0) for key in criteria]
        self.last_confidence = max(confidences) if confidences else None
        return scores


class LayaTreeBackend(LayaBackend):
    """Laya driving a hierarchical decision tree.

    Defaults to the `typed-decisions` checkpoint, which is the one fine-tuned for
    structured typed questions (0.766 accuracy on its benchmark against 0.362 for
    the English base checkpoint) and which natively runs the 1024/256 budget that
    the extra tree context needs.
    """

    is_tree = True

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        subfolder: Optional[str] = "typed-decisions",
        **kwargs,
    ):
        super().__init__(model_name=model_name, subfolder=subfolder, **kwargs)


class ApiLlmBackend(BaseBackend):
    """API LLM backend for automated benchmarking (OpenAI, Anthropic, Gemini, OpenRouter, Ollama)."""

    wants_full_context = True

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 45.0,
        tree: bool = False,
    ):
        super().__init__()
        self.model = model
        self.timeout = timeout
        self.is_tree = tree

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

        self.calls += 1
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

            parsed = json.loads(content)
            choice = parsed.get("choice", "")
            if choice in criteria:
                return choice

            for k in valid_keys:
                if f'"{k}"' in content or f"'{k}'" in content or re.search(rf"\b{re.escape(k)}\b", content):
                    return k
        except Exception:
            pass

        self.note_fallback()
        return valid_keys[0]


class AgenticBackend(BaseBackend):
    """Agentic AI backend: an external agent loop, a test hook, or a human at the console."""

    wants_full_context = True

    def __init__(
        self,
        decision_hook: Optional[Callable[[str, str, str, Dict[str, str]], str]] = None,
        interactive: bool = False,
        **kwargs,
    ):
        super().__init__()
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

        self.calls += 1

        if self.decision_hook:
            try:
                res = self.decision_hook(context, question_id, instructions, criteria)
                if res in criteria:
                    return res
            except Exception:
                pass

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

        self.note_fallback()
        return valid_keys[0]


def create_backend(
    backend_type: str = "laya",
    model: Optional[str] = None,
    **kwargs,
) -> BaseBackend:
    """Factory function to instantiate the requested backend."""
    b_type = backend_type.lower()
    if b_type in ("laya", "decision_model"):
        return LayaBackend(model_name=model or "convaiinnovations/laya", **kwargs)
    elif b_type in ("laya_tree", "tree", "hierarchical"):
        return LayaTreeBackend(model_name=model or "convaiinnovations/laya", **kwargs)
    elif b_type in ("api", "api_llm", "llm"):
        return ApiLlmBackend(model=model or "gpt-4o", **kwargs)
    elif b_type in ("api_llm_tree", "llm_tree"):
        return ApiLlmBackend(model=model or "gpt-4o", tree=True, **kwargs)
    elif b_type in ("agentic", "interactive", "manual"):
        return AgenticBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. Choose 'laya', 'laya_tree', "
            f"'api_llm', 'api_llm_tree', or 'agentic'."
        )
