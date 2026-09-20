"""Stress harness: how far does a decision policy get in the offline simulator?

The probe policies below are MEASURING INSTRUMENTS, not players.

  `first`  -- always takes the first option (what Python's own fallback does)
  `random` -- uniform random, the floor any real model must clear
  `label`  -- a perfect reader of the option text, and nothing else

`label` is the important one. It sees exactly what Laya sees -- the same option
strings, the same brief -- and nothing more. It therefore answers the design
question directly: *if a model understood our labels perfectly, could it win?*
If `label` cannot survive Ante 1, the tree is not presenting enough information
and no model will save it. If `label` does well and Laya does not, the gap is the
model's, and the tree is doing its job.

None of this may ever be imported by the live runner: wiring `label` into play
would make Python the player. `tests/test_decision_boundary.py` enforces that.
"""

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sidecar.agent import BalatroAgent
from sidecar.backends import BaseBackend
from sidecar.sim import JOKER_LIBRARY, BalatroSim, RunResult, SimJoker

_CHIPS_RE = re.compile(r"~(\d+)\s*chips")


class ProbeBackend(BaseBackend):
    """A scripted answerer used only for measurement. Never for play."""

    def __init__(self, policy: str = "first", seed: int = 0):
        super().__init__()
        self.policy = policy
        import random as _random
        self.rng = _random.Random(seed)
        self._last_context = ""

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _chips(text: str) -> int:
        match = _CHIPS_RE.search(text or "")
        return int(match.group(1)) if match else -1

    def predict_choice(self, context, question_id, instructions, criteria):
        self.calls += 1
        self._last_context = context
        keys = list(criteria)
        if not keys:
            return ""
        if self.policy == "random":
            return self.rng.choice(keys)
        if self.policy == "label":
            return self._label_choice(question_id, criteria, keys)
        return keys[0]

    def predict_bool(self, context, question_id, statement, true_desc="", false_desc=""):
        self.calls += 1
        if self.policy == "random":
            return self.rng.random() < 0.5
        if self.policy == "label":
            return self._label_bool(context, question_id)
        return False

    def rank_options(self, context, question_id, statement_for, criteria):
        """Probes answer polls by scoring the option they would have chosen."""
        chosen = self.predict_choice(context, question_id, "", criteria)
        return {key: (1.0 if key == chosen else 0.0) for key in criteria}

    # -- the label-reading probe ----------------------------------------
    def _label_choice(self, question_id: str, criteria: Dict[str, str], keys: List[str]) -> str:
        if question_id in ("shop_category", "which_shop_item", "shop_action", "pack_choice",
                           "consumable_action", "sell_joker_for_room", "joker_order"):
            return keys[0]
        if question_id == "which_discard":
            return self._best_discard(criteria, keys)
        if question_id == "hand_action":
            # Flat mode puts plays and discards in one question, so the probe has
            # to make the dig call here rather than at a dedicated node.
            discards = {k: v for k, v in criteria.items() if k.startswith("discard_")}
            plays = {k: v for k, v in criteria.items() if not k.startswith("discard_")}
            if discards and plays and self._label_bool(self._last_context, "dig_for_better_hand"):
                return self._best_discard(discards, list(discards))
            criteria = plays or criteria
            keys = list(criteria)
        scored = [(self._chips(text), key) for key, text in criteria.items()]
        best = max(scored)
        return best[1] if best[0] >= 0 else keys[0]

    @staticmethod
    def _best_discard(criteria: Dict[str, str], keys: List[str]) -> str:
        """Prefer the draw that is one card short of a made hand, then kept pairs."""
        for wanted in ("short of a Flush", "short of a Straight", "keep the"):
            for key, text in criteria.items():
                if wanted in text:
                    return key
        return keys[0]

    def _label_bool(self, context: str, question_id: str) -> bool:
        if question_id == "send_filler_cards":
            return True
        if question_id == "skip_blind":
            return False
        if question_id != "dig_for_better_hand":
            return False
        # Dig while the listed hands cannot finish the blind and digging is affordable.
        if "WINS" in context:
            return False
        if "discarding lowers your score" in context:
            return False
        hands_left = self._int_after(context, r"(\d+) hands and")
        discards_left = self._int_after(context, r"and (\d+) discards")
        if hands_left <= 1 or discards_left <= 0:
            return False
        return "still lose" in context

    @staticmethod
    def _int_after(text: str, pattern: str) -> int:
        match = re.search(pattern, text or "")
        return int(match.group(1)) if match else 0


@dataclass
class StressReport:
    policy: str
    mode: str
    node: str = "poll"
    runs: List[RunResult] = field(default_factory=list)
    decisions: int = 0
    fallbacks: int = 0

    @property
    def rounds_won(self) -> List[int]:
        return [r.rounds_won for r in self.runs]

    @property
    def ante1_clear_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.rounds_won >= 3) / len(self.runs)

    @property
    def first_blind_clear_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.rounds_won >= 1) / len(self.runs)

    def summary(self) -> str:
        won = self.rounds_won or [0]
        return (
            f"{self.policy:<8} {self.mode:<5} {self.node:<6} "
            f"rounds won: mean {statistics.mean(won):5.2f} max {max(won):2d} | "
            f"beat 1st blind {self.first_blind_clear_rate * 100:5.1f}% | "
            f"cleared Ante 1 {self.ante1_clear_rate * 100:5.1f}% | "
            f"decisions {self.decisions:5d} fallbacks {self.fallbacks}"
        )


def build_backend(policy: str, seed: int, tree: bool) -> BaseBackend:
    if policy == "laya":
        from sidecar.backends import LayaBackend, LayaTreeBackend
        return LayaTreeBackend() if tree else LayaBackend()
    backend = ProbeBackend(policy=policy, seed=seed)
    backend.is_tree = tree
    return backend


def run_stress(
    policy: str = "label",
    mode: str = "tree",
    seeds: Sequence[int] = range(20),
    jokers: Optional[Sequence[SimJoker]] = None,
    max_ante: int = 3,
    shared_backend: bool = True,
    node: str = "poll",
) -> StressReport:
    tree = mode == "tree"
    report = StressReport(policy=policy, mode=mode, node=node)
    backend = build_backend(policy, 0, tree) if shared_backend else None
    for seed in seeds:
        active = backend or build_backend(policy, seed, tree)
        agent = BalatroAgent(backend=active, tree_mode=tree, poll_mode=(node == "poll" and tree))
        sim = BalatroSim(seed=seed, jokers=jokers)
        report.runs.append(sim.play_run(agent, max_ante=max_ante))
    final = backend or active
    report.decisions = final.calls
    report.fallbacks = final.fallbacks
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test Balatro decision trees offline.")
    parser.add_argument("--policy", default="label",
                        choices=["label", "first", "random", "laya"],
                        help="Probe policy, or 'laya' for the real decision model")
    parser.add_argument("--mode", default="both", choices=["tree", "flat", "both"])
    parser.add_argument("--node", default="poll", choices=["poll", "choice"],
                        help="Tree node shape: batched yes/no poll, or one wide choice")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-ante", type=int, default=3)
    parser.add_argument("--jokers", default="", help="Comma-separated keys from JOKER_LIBRARY")
    parser.add_argument("--verbose", action="store_true", help="Print every action of seed 0")
    args = parser.parse_args()

    jokers = [JOKER_LIBRARY[k.strip()] for k in args.jokers.split(",") if k.strip() in JOKER_LIBRARY]
    modes = ["tree", "flat"] if args.mode == "both" else [args.mode]

    joker_note = ", ".join(j.name for j in jokers) if jokers else "none"
    print(f"Stress: {args.seeds} seeds, up to Ante {args.max_ante}, jokers: {joker_note}\n")
    for mode in modes:
        report = run_stress(policy=args.policy, mode=mode, node=args.node,
                            seeds=range(args.seeds), jokers=jokers, max_ante=args.max_ante)
        print(report.summary())
        if args.verbose and report.runs:
            print("\n  seed 0 trace:")
            for rnd in report.runs[0].rounds:
                verdict = "WON " if rnd.won else "LOST"
                print(f"   {verdict} Ante {rnd.ante} {rnd.blind}: {rnd.chips}/{rnd.target}")
                for act in rnd.actions:
                    print(f"       {act}")
            print()


if __name__ == "__main__":
    main()
