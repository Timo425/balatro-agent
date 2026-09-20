"""Test doubles shared by the suite."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from sidecar.backends import BaseBackend


class ScriptedBackend(BaseBackend):
    """Answers questions from a script and records every question it was asked.

    Used to assert that the agent honours whatever the model answers -- that is
    the property that keeps Python out of the driver's seat.
    """

    def __init__(self, answers: Optional[Dict[str, Any]] = None, tree: bool = True):
        super().__init__()
        self.answers: Dict[str, Any] = dict(answers or {})
        self.is_tree = tree
        self.log: List[Tuple[str, Dict[str, str]]] = []

    def asked(self, question_id: str) -> List[Dict[str, str]]:
        return [criteria for qid, criteria in self.log if qid == question_id]

    @property
    def question_ids(self) -> List[str]:
        return [qid for qid, _ in self.log]

    def predict_choice(self, context, question_id, instructions, criteria):
        self.calls += 1
        self.log.append((question_id, dict(criteria)))
        answer = self.answers.get(question_id)
        if callable(answer):
            answer = answer(criteria)
        if answer in criteria:
            return answer
        return next(iter(criteria))

    def predict_bool(self, context, question_id, statement, true_desc="", false_desc=""):
        self.calls += 1
        self.log.append((question_id, {"true": true_desc, "false": false_desc}))
        answer = self.answers.get(question_id)
        if callable(answer):
            answer = answer(statement)
        return bool(answer)


class BrokenBackend(BaseBackend):
    """A backend that always fails, to exercise the fallback path."""

    def __init__(self, tree: bool = True):
        super().__init__()
        self.is_tree = tree

    def predict_choice(self, context, question_id, instructions, criteria):
        raise RuntimeError("backend unavailable")

    def predict_bool(self, context, question_id, statement, true_desc="", false_desc=""):
        raise RuntimeError("backend unavailable")
