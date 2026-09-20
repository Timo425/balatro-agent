"""Question benchmark: does the decision model see anything in Balatro at all?

A stress run says how far a model gets. It does not say *why* it fails, because
every round mixes dozens of judgement calls with luck. This file asks a different
question: on decision points where there is a provably correct answer, how often
does the model give it?

Each item is generated from a real dealt hand, in exactly the format the live
tree uses, and carries a ground truth the simulator can verify. Two variants of
every item are produced:

  `labelled` -- the real thing: `WINS - Flush AS KS QS JS 9S: ~340 chips, wins now`
  `bare`     -- the same options with every number and verdict stripped:
                `Flush AS KS QS JS 9S`

The gap between them is the interesting number. If a model scores well on
`labelled` and at chance on `bare`, it can read a verdict but not derive one --
which is precisely what a System 1 pattern matcher is supposed to do, and
precisely why the facts have to be computed for it.

MEASUREMENT ONLY. This module knows correct answers, so it must never be
imported by the play path; tests/test_decision_boundary.py enforces that.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sidecar.backends import BaseBackend
from sidecar.candidates import PlayOption, annotate_play_options, enumerate_play_cores
from sidecar.poker_eval import Card
from sidecar.sim import BalatroSim

_LABEL_NOISE = re.compile(
    r"(WINS|STRONG|OK|WEAK|TINY)\s*-\s*|:\s*~\d+\s*chips.*$|,\s*~\d+\s*chips.*$"
)


def strip_labels(text: str) -> str:
    """Removes every verdict and number, leaving the cards themselves."""
    return _LABEL_NOISE.sub("", text).strip(" ;:,")


@dataclass
class QItem:
    kind: str
    variant: str
    context: str
    instructions: str
    criteria: Dict[str, str]
    answer: Any
    qtype: str = "choice"


@dataclass
class QResult:
    items: int = 0
    correct: int = 0
    confidences: List[float] = field(default_factory=list)
    hits: List[int] = field(default_factory=list)
    given: List[Any] = field(default_factory=list)

    @property
    def top_answer_share(self) -> float:
        """Fraction taken by the most common answer.

        A model that answers the same thing whatever the board scores 1.00 here,
        and its accuracy is then just the base rate of that answer. This is the
        number that separates "gets it wrong" from "is not reading the question".
        """
        if not self.given:
            return float("nan")
        counts: Dict[Any, int] = defaultdict(int)
        for g in self.given:
            counts[g] += 1
        return max(counts.values()) / len(self.given)

    @property
    def accuracy(self) -> float:
        return self.correct / self.items if self.items else 0.0

    @property
    def mean_confidence(self) -> float:
        vals = [c for c in self.confidences if c is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def ece(self) -> float:
        """Expected calibration error over 10 bins."""
        pairs = [(c, h) for c, h in zip(self.confidences, self.hits) if c is not None]
        if not pairs:
            return float("nan")
        total = 0.0
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            chunk = [(c, h) for c, h in pairs if lo < c <= hi or (i == 0 and c == 0)]
            if chunk:
                conf = sum(c for c, _ in chunk) / len(chunk)
                acc = sum(h for _, h in chunk) / len(chunk)
                total += len(chunk) / len(pairs) * abs(conf - acc)
        return total


def _deal(seed: int, count: int = 8) -> List[Card]:
    sim = BalatroSim(seed=seed)
    deck = sim.fresh_deck()
    hand = deck[:count]
    for i, card in enumerate(hand):
        card.id = i
    return hand


def _options(hand: Sequence[Card], hands_left: int) -> List[PlayOption]:
    return annotate_play_options(enumerate_play_cores(hand), None, [], hands_left=hands_left)


def _brief(hand: Sequence[Card], needed: int, hands_left: int, discards_left: int,
           options: Sequence[PlayOption], labelled: bool) -> str:
    lines = [
        "Balatro. Ante 1, Small Blind.",
        f"Need {needed} more chips. {hands_left} hands and {discards_left} discards left.",
        "Jokers: none",
        "Cards in hand: " + " ".join(c.short for c in hand),
        "Hands you can make right now:",
    ]
    for opt in options:
        text = opt.label(needed, hands_left)
        lines.append("- " + (text if labelled else strip_labels(text)))
    return "\n".join(lines)


def _shuffled(criteria: Dict[str, str], seed: int) -> Dict[str, str]:
    """Deterministically reorders options so position carries no signal.

    The live tree lists options in the game's fixed hand ranking, where the
    biggest hand usually comes first. A benchmark that kept that order would hand
    full marks to any model that simply picks option one, which is the opposite
    of what is being measured.
    """
    keys = list(criteria)
    random.Random(seed).shuffle(keys)
    return {k: criteria[k] for k in keys}


def build_items(seeds: Sequence[int]) -> List[QItem]:
    """Generates decision items with verifiable answers, labelled and bare."""
    items: List[QItem] = []
    for seed in seeds:
        hand = _deal(seed)
        options = _options(hand, hands_left=1)
        if len(options) < 2:
            continue
        best = max(options, key=lambda o: o.estimate.total)
        worst = min(options, key=lambda o: o.estimate.total)
        if best.key == worst.key:
            continue

        for labelled in (True, False):
            variant = "labelled" if labelled else "bare"

            # 1. Two hands, last hand of the round: the bigger one is correct.
            needed = best.estimate.total  # exactly reachable by the better hand
            pair_options = [best, worst]
            criteria = {}
            for opt in pair_options:
                text = opt.label(needed, 1)
                criteria[opt.key] = text if labelled else strip_labels(text)
            items.append(QItem(
                kind="bigger_of_two",
                variant=variant,
                context=_brief(hand, needed, 1, 0, pair_options, labelled),
                instructions="This is the last hand of the round. "
                             "Pick the hand that scores the most chips.",
                criteria=_shuffled(criteria, seed),
                answer=best.key,
            ))

            # 2. The full option list, same question.
            criteria_all = {}
            for opt in options:
                text = opt.label(needed, 1)
                criteria_all[opt.key] = text if labelled else strip_labels(text)
            items.append(QItem(
                kind="biggest_of_all",
                variant=variant,
                context=_brief(hand, needed, 1, 0, options, labelled),
                instructions="This is the last hand of the round. "
                             "Pick the hand that scores the most chips.",
                criteria=_shuffled(criteria_all, seed + 1),
                answer=best.key,
            ))

            # 3. Can this hand finish the blind? Both answers appear equally often.
            for factor, truth in ((0.5, True), (2.0, False)):
                target = max(1, int(best.estimate.total * factor))
                opts = _options(hand, hands_left=1)
                items.append(QItem(
                    kind="beats_the_blind",
                    variant=variant,
                    qtype="noul",
                    context=_brief(hand, target, 1, 0, opts, labelled),
                    instructions="Can the player reach the chips this blind needs "
                                 "with one of the hands listed?",
                    criteria={"true": "yes, one of them is enough",
                              "false": "no, none of them is enough"},
                    answer=truth,
                ))

            # 4. Play or dig, constructed so the answer is not arguable:
            #    a hand that already wins (play) against one that cannot get a
            #    tenth of the way there with four hands (dig).
            for target, truth in ((max(1, best.estimate.total // 2), False),
                                  (best.estimate.total * 40, True)):
                opts = _options(hand, hands_left=4)
                items.append(QItem(
                    kind="dig_or_play",
                    variant=variant,
                    qtype="noul",
                    context=_brief(hand, target, 4, 4, opts, labelled),
                    instructions="Should the player throw cards away and draw new "
                                 "ones instead of playing one of the hands listed?",
                    criteria={"true": "throw cards away now and draw replacements",
                              "false": "play one of the hands listed above now"},
                    answer=truth,
                ))
    return items


def evaluate(backend: BaseBackend, items: Sequence[QItem]) -> Dict[Tuple[str, str], QResult]:
    results: Dict[Tuple[str, str], QResult] = defaultdict(QResult)
    for item in items:
        if item.qtype == "noul":
            given = backend.predict_bool(item.context, item.kind, item.instructions,
                                         item.criteria.get("true", ""),
                                         item.criteria.get("false", ""))
        else:
            given = backend.predict_choice(item.context, item.kind,
                                           item.instructions, item.criteria)
        hit = int(given == item.answer)
        bucket = results[(item.kind, item.variant)]
        bucket.items += 1
        bucket.correct += hit
        bucket.confidences.append(backend.last_confidence)
        bucket.hits.append(hit)
        bucket.given.append(given)
    return results


def chance_level(items: Sequence[QItem], kind: str, variant: str) -> float:
    picked = [i for i in items if i.kind == kind and i.variant == variant]
    if not picked:
        return float("nan")
    return sum(1.0 / len(i.criteria) for i in picked) / len(picked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a decision model on Balatro questions with known answers.")
    parser.add_argument("--backend", default="laya_tree",
                        choices=["laya", "laya_tree", "first", "random"])
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--json", default=None, help="Write the raw table to this path")
    args = parser.parse_args()

    items = build_items(range(args.seeds))
    if args.backend in ("first", "random"):
        from sidecar.stress import ProbeBackend
        backend: BaseBackend = ProbeBackend(policy=args.backend, seed=0)
    else:
        from sidecar.backends import create_backend
        backend = create_backend(args.backend)

    results = evaluate(backend, items)

    print(f"\n{args.backend} on {len(items)} items from {args.seeds} dealt hands\n")
    print(f"{'question':<18} {'variant':<9} {'n':>4} {'accuracy':>9} {'chance':>7} "
          f"{'same answer':>12} {'confidence':>11} {'ECE':>6}")
    print("-" * 84)
    rows = []
    for kind in ("bigger_of_two", "biggest_of_all", "beats_the_blind", "dig_or_play"):
        for variant in ("labelled", "bare"):
            res = results.get((kind, variant))
            if not res or not res.items:
                continue
            chance = chance_level(items, kind, variant)
            print(f"{kind:<18} {variant:<9} {res.items:>4} {res.accuracy:>8.1%} "
                  f"{chance:>7.1%} {res.top_answer_share:>11.1%} "
                  f"{res.mean_confidence:>11.3f} {res.ece:>6.3f}")
            rows.append({"question": kind, "variant": variant, "n": res.items,
                         "accuracy": res.accuracy, "chance": chance,
                         "top_answer_share": res.top_answer_share,
                         "mean_confidence": res.mean_confidence, "ece": res.ece})
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"backend": args.backend, "seeds": args.seeds, "rows": rows}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
