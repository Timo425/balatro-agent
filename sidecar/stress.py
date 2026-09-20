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
import json
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
        if question_id == "which_blind_next":
            return "play_this_blind"
        if question_id == "which_aim":
            return self._label_aim(criteria, keys)
        if set(keys) == {"yes", "no"}:
            # The same decision asked as a two-option choice.
            return "yes" if self._label_bool(self._last_context, question_id) else "no"
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

    def _label_aim(self, criteria: Dict[str, str], keys: List[str]) -> str:
        """Play now, or dig, read off the same option text the model sees."""
        digs = {k: v for k, v in criteria.items() if k.startswith("dig_")}
        plays = {k: v for k, v in criteria.items() if not k.startswith("dig_")}
        winning = [k for k, v in plays.items() if "WINS" in v]
        if winning:
            return max(winning, key=lambda k: self._chips(plays[k]))
        if digs and self._label_bool(self._last_context, "dig_for_better_hand"):
            return self._best_dig(digs, list(digs))
        if plays:
            return max(plays, key=lambda k: self._chips(plays[k]))
        return keys[0]

    @staticmethod
    def _best_dig(criteria: Dict[str, str], keys: List[str]) -> str:
        for wanted in ("Flush", "Straight", "Full House", "Four of a Kind",
                       "Three of a Kind", "jokers"):
            for key, text in criteria.items():
                if wanted in text:
                    return key
        return keys[0]

    @staticmethod
    def _best_discard(criteria: Dict[str, str], keys: List[str]) -> str:
        """Prefer the draw that is one card short of a made hand, then kept pairs."""
        for wanted in ("Flush", "Straight", "keep the"):
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


class RecordingBackend(BaseBackend):
    """Wraps a backend and writes every question and answer to a JSONL log.

    It changes no decision -- it forwards each call and records what came back.
    The point is the training data: a stress run produces (state, question,
    options, answer) rows, and the simulator knows afterwards whether the round
    was won, which is exactly the reward signal a fine-tune of a decision model
    needs. Laya's own model card is blunt that its base checkpoints are near
    chance on typed decisions until fine-tuned on the target workflow.
    """

    def __init__(self, inner: BaseBackend):
        super().__init__()
        self.inner = inner
        self.is_tree = inner.is_tree
        self.wants_full_context = inner.wants_full_context
        self.rows: List[Dict[str, Any]] = []
        self.open_rows: List[Dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return self.inner.calls

    @calls.setter
    def calls(self, value: int) -> None:
        if hasattr(self, "inner"):
            self.inner.calls = value

    @property
    def fallbacks(self) -> int:
        return self.inner.fallbacks

    @fallbacks.setter
    def fallbacks(self, value: int) -> None:
        if hasattr(self, "inner"):
            self.inner.fallbacks = value

    def _record(self, context, question_id, kind, options, answer):
        row = {
            "question_id": question_id,
            "type": kind,
            "context": context,
            "options": options,
            "answer": answer,
            "confidence": self.inner.last_confidence,
        }
        self.rows.append(row)
        self.open_rows.append(row)
        return answer

    def close_round(self, seed: int, ante: int, blind: str, won: bool, ratio: float) -> None:
        """Stamps the rows since the last close with how that round turned out."""
        for row in self.open_rows:
            row.update(seed=seed, ante=ante, blind=blind, won=won, score_ratio=round(ratio, 3))
        self.open_rows = []

    def predict_choice(self, context, question_id, instructions, criteria):
        answer = self.inner.predict_choice(context, question_id, instructions, criteria)
        return self._record(context, question_id, "choice", dict(criteria), answer)

    def predict_bool(self, context, question_id, statement, true_desc="", false_desc=""):
        answer = self.inner.predict_bool(context, question_id, statement, true_desc, false_desc)
        return self._record(context, question_id, "noul",
                            {"true": true_desc, "false": false_desc}, answer)

    def rank_options(self, context, question_id, statement_for, criteria):
        scores = self.inner.rank_options(context, question_id, statement_for, criteria)
        best = max(scores, key=lambda k: scores[k]) if scores else None
        self._record(context, question_id, "poll", dict(criteria), best)
        return scores


@dataclass
class StressReport:
    policy: str
    mode: str
    node: str = "choice"
    runs: List[RunResult] = field(default_factory=list)
    decisions: int = 0
    fallbacks: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def illegal_moves(self) -> int:
        return sum(len(rnd.illegal) for run in self.runs for rnd in run.rounds)

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
            f"decisions {self.decisions:5d} fallbacks {self.fallbacks} "
            f"illegal {self.illegal_moves}"
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
    node: str = "choice",
    record: bool = False,
    binary: str = "noul",
) -> StressReport:
    tree = mode == "tree"
    report = StressReport(policy=policy, mode=mode, node=node)
    backend = build_backend(policy, 0, tree) if shared_backend else None
    if record and backend is not None:
        backend = RecordingBackend(backend)
    for seed in seeds:
        active = backend or build_backend(policy, seed, tree)
        agent = BalatroAgent(backend=active, tree_mode=tree,
                             poll_mode=(node == "poll" and tree), binary_nodes=binary)
        sim = BalatroSim(seed=seed, jokers=jokers)
        run = sim.play_run(agent, max_ante=max_ante)
        report.runs.append(run)
        if isinstance(active, RecordingBackend):
            for rnd in run.rounds:
                active.close_round(seed, rnd.ante, rnd.blind, rnd.won,
                                   rnd.chips / max(1, rnd.target))
    final = backend or active
    report.decisions = final.calls
    report.fallbacks = final.fallbacks
    report.rows = getattr(final, "rows", [])
    return report


def compare(policy: str, mode: str, seeds: Sequence[int], jokers, max_ante: int,
            node: str, binary: str) -> Dict[str, StressReport]:
    """Runs the policy, random and the label reader over the same seeds.

    The gap between the policy and `random` is how much the model is actually
    contributing. If a change to the tree makes that gap shrink towards zero
    while the score stays up, the tree has started playing the game itself and
    the model is only rubber-stamping it -- which is the failure this repository
    exists to avoid, and it is invisible unless measured.
    """
    return {
        name: run_stress(policy=name, mode=mode, seeds=seeds, jokers=jokers,
                         max_ante=max_ante, node=node, binary=binary)
        for name in (policy, "random", "label")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test Balatro decision trees offline.")
    parser.add_argument("--policy", default="label",
                        choices=["label", "first", "random", "laya"],
                        help="Probe policy, or 'laya' for the real decision model")
    parser.add_argument("--mode", default="both", choices=["tree", "flat", "both"])
    parser.add_argument("--node", default="choice", choices=["choice", "poll"],
                        help="Tree node shape: one choice per node, or a batched yes/no poll")
    parser.add_argument("--binary", default="noul", choices=["noul", "choice"],
                        help="How yes/no nodes are asked: natively, or as a two-option choice")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-ante", type=int, default=3)
    parser.add_argument("--jokers", default="", help="Comma-separated keys from JOKER_LIBRARY")
    parser.add_argument("--verbose", action="store_true", help="Print every action of seed 0")
    parser.add_argument("--compare", action="store_true",
                        help="Also run random and the label reader over the same seeds, and "
                             "report how much the policy is contributing over random")
    parser.add_argument("--dump", default=None, metavar="PATH",
                        help="Write every question, answer and round outcome to a JSONL "
                             "file -- reward-labelled data for fine-tuning a decision model")
    args = parser.parse_args()

    jokers = [JOKER_LIBRARY[k.strip()] for k in args.jokers.split(",") if k.strip() in JOKER_LIBRARY]
    modes = ["tree", "flat"] if args.mode == "both" else [args.mode]

    joker_note = ", ".join(j.name for j in jokers) if jokers else "none"
    print(f"Stress: {args.seeds} seeds, up to Ante {args.max_ante}, jokers: {joker_note}\n")
    for mode in modes:
        if args.compare:
            reports = compare(args.policy, mode, range(args.seeds), jokers,
                              args.max_ante, args.node, args.binary)
            for name in reports:
                print(reports[name].summary())
            policy_mean = statistics.mean(reports[args.policy].rounds_won or [0])
            random_mean = statistics.mean(reports["random"].rounds_won or [0])
            label_mean = statistics.mean(reports["label"].rounds_won or [0])
            gap = policy_mean - random_mean
            headroom = label_mean - policy_mean
            print(f"\n  {args.policy} over random: {gap:+.2f} rounds"
                  f"   (headroom to a perfect label reader: {headroom:+.2f})")
            if gap <= 0.1:
                print("  ** the policy is not beating random: either the model is not "
                      "answering, or the tree is playing for it **")
            print()
            report = reports[args.policy]
        else:
            report = run_stress(policy=args.policy, mode=mode, node=args.node,
                                seeds=range(args.seeds), jokers=jokers, max_ante=args.max_ante,
                                record=bool(args.dump), binary=args.binary)
            print(report.summary())
        if args.dump:
            path = Path(args.dump)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                for row in report.rows:
                    handle.write(json.dumps({**row, "mode": mode, "node": args.node,
                                             "policy": args.policy}) + "\n")
            print(f"  wrote {len(report.rows)} decisions to {path}")
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
