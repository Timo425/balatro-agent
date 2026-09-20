"""The invariant that keeps Python out of the driver's seat.

Every option the agent offers must be reachable: if the model names it, the agent
must carry it out unchanged. If any option were quietly overridden, filtered or
re-ranked by Python, one of these tests would fail.
"""

import ast
import unittest
from pathlib import Path

from sidecar.agent import BalatroAgent, BalatroGameState
from sidecar.poker_eval import Card
from tests.support import BrokenBackend, ScriptedBackend

ROOT = Path(__file__).resolve().parent.parent
PLAY_PATH_MODULES = ["agent.py", "runner.py", "client.py", "bench.py", "cli.py",
                     "candidates.py", "scoring.py", "brief.py", "backends.py",
                     "poker_eval.py", "synergy.py", "strategy.py"]


def make_state(**overrides):
    hand = overrides.pop("hand", [
        Card("Q", "Hearts", 0), Card("9", "Spades", 1), Card("9", "Diamonds", 2),
        Card("8", "Hearts", 3), Card("8", "Diamonds", 4), Card("7", "Diamonds", 5),
        Card("6", "Diamonds", 6), Card("3", "Clubs", 7),
    ])
    base = dict(ante=1, blind_name="Small Blind", target_chips=300, current_chips=0,
                hands_left=4, discards_left=4, money=10, hand=hand, jokers=[])
    base.update(overrides)
    return BalatroGameState(**base)


class TestEveryOptionIsReachable(unittest.TestCase):
    def offered(self, question_id, answers=None, tree=True):
        backend = ScriptedBackend(answers or {}, tree=tree)
        agent = BalatroAgent(backend=backend, tree_mode=tree, poll_mode=False)
        agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        asked = backend.asked(question_id)
        self.assertTrue(asked, f"{question_id} was never asked")
        return asked[0]

    def aim_options(self, kind):
        """`kind` is "dig" for the discard plans, "play" for everything else."""
        keys = list(self.offered("which_aim"))
        if kind == "dig":
            return [k for k in keys if k.startswith("dig_")]
        return [k for k in keys if not k.startswith("dig_")]

    def test_each_playable_hand_is_played_when_chosen(self):
        keys = self.aim_options("play")
        self.assertGreater(len(keys), 1)
        seen = set()
        for key in keys:
            backend = ScriptedBackend({"which_aim": key, "send_filler_cards": False})
            agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
            action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
            self.assertEqual(action.action_type, "PLAY", key)
            seen.add(tuple(sorted(c.id for c in action.cards)))
        self.assertEqual(len(seen), len(keys),
                         "two different hand types produced the same move")

    def test_each_hand_of_a_type_is_reachable_when_grouped(self):
        """On the tighter budget the tree groups by type, and the second question
        must still be able to reach every hand inside the chosen group."""
        from sidecar.candidates import enumerate_play_cores

        by_key = {o.key: o for o in enumerate_play_cores(make_state().hand)}

        def agent_for(answers):
            backend = ScriptedBackend(answers, head_max_len=192)
            return BalatroAgent(backend=backend, tree_mode=True, poll_mode=False), backend

        agent, probe = agent_for({"which_aim": "play_pair", "send_filler_cards": False})
        agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        offered = probe.asked("which_hand_of_type")[0]
        self.assertGreater(len(offered), 1)
        for key in offered:
            agent, _ = agent_for({"which_aim": "play_pair", "which_hand_of_type": key,
                                  "send_filler_cards": False})
            action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
            played = {c.id for c in action.cards}
            self.assertTrue({c.id for c in by_key[key].core} <= played, key)

    def test_each_dig_plan_is_executed_when_chosen(self):
        keys = self.aim_options("dig")
        self.assertGreater(len(keys), 1)
        seen = set()
        for key in keys:
            backend = ScriptedBackend({"which_aim": key})
            agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
            action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
            self.assertEqual(action.action_type, "DISCARD", key)
            seen.add(tuple(sorted(c.id for c in action.cards)))
        self.assertEqual(len(seen), len(keys))

    def test_the_aim_answer_decides_play_versus_discard(self):
        dig_key = self.aim_options("dig")[0]
        play_key = self.aim_options("play")[0]
        for key, expected in ((dig_key, "DISCARD"), (play_key, "PLAY")):
            backend = ScriptedBackend({"which_aim": key, "send_filler_cards": False})
            agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
            action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
            self.assertEqual(action.action_type, expected)

    def test_filler_answer_changes_the_cards_played(self):
        without = ScriptedBackend({"which_aim": "pair_9", "send_filler_cards": False})
        with_filler = ScriptedBackend({"which_aim": "pair_9", "send_filler_cards": True})
        a = BalatroAgent(backend=without, tree_mode=True, poll_mode=False).decide_hand_action(
            make_state(), raw_g={"hands": {}})
        b = BalatroAgent(backend=with_filler, tree_mode=True, poll_mode=False).decide_hand_action(
            make_state(), raw_g={"hands": {}})
        self.assertEqual(len(a.cards), 2)
        self.assertEqual(len(b.cards), 5)

    def test_poll_scores_decide_the_hand(self):
        """With polling, the option the model scores highest is the one played."""
        for wanted in ("pair_9", "pair_8"):
            backend = ScriptedBackend({"which_aim": "play_pair",
                                       "send_filler_cards": False}, head_max_len=192)
            backend.rank_options = (
                lambda ctx, qid, stmt, crit, w=wanted:
                {k: (1.0 if k == w else 0.1) for k in crit}
            )
            agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=True)
            action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
            self.assertEqual(action.action_type, "PLAY")
            self.assertTrue(action.reasoning.startswith("Tree"), action.reasoning)

    def test_shop_every_item_is_buyable(self):
        shop = [{"name": "Joker", "cost": 4, "set": "JOKER"},
                {"name": "Banner", "cost": 5, "set": "JOKER"},
                {"name": "Voucher: Clearance Sale", "cost": 10, "set": "VOUCHER"}]
        state = make_state(money=20, hand=[])
        for idx in range(len(shop)):
            key_prefix = f"buy{idx}_"
            backend = ScriptedBackend({
                "shop_category": lambda crit: next(iter(crit)),
                "which_shop_item": lambda crit, p=key_prefix: next(
                    (k for k in crit if k.startswith(p)), next(iter(crit))),
                "shop_action": lambda crit, p=key_prefix: next(
                    (k for k in crit if k.startswith(p)), next(iter(crit))),
            })
            agent = BalatroAgent(backend=backend, tree_mode=False, poll_mode=False)
            decision = agent.decide_shop_action(state, shop, raw_g={"jokers": {"cards": []}})
            self.assertIsNotNone(decision, f"item {idx} unreachable")
            self.assertEqual(decision[1], idx)

    def test_shop_leave_answer_buys_nothing(self):
        shop = [{"name": "Joker", "cost": 4, "set": "JOKER"}]
        backend = ScriptedBackend({"shop_category": "cat_save", "shop_action": "leave_shop"})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        self.assertIsNone(agent.decide_shop_action(make_state(hand=[]), shop,
                                                   raw_g={"jokers": {"cards": []}}))

    def test_blind_skip_answer_is_honoured(self):
        raw_g = {"blinds": {"small": {"status": "SELECT", "name": "Small Blind",
                                      "tag_name": "Negative Tag", "chips": 300}}}
        for answer, expected in (("skip_to_next_blind", "SKIP"),
                                 ("play_this_blind", "SELECT")):
            backend = ScriptedBackend({"which_blind_next": answer})
            agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
            action, _ = agent.decide_blind_action(make_state(hand=[]), raw_g)
            self.assertEqual(action, expected)


class TestFallbacksAreVisible(unittest.TestCase):
    def test_broken_backend_is_counted_and_labelled(self):
        backend = BrokenBackend()
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        self.assertIn("FALLBACK", action.reasoning)
        self.assertGreater(backend.fallbacks, 0)

    def test_broken_backend_buys_nothing(self):
        backend = BrokenBackend()
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        shop = [{"name": "Joker", "cost": 4, "set": "JOKER"}]
        self.assertIsNone(agent.decide_shop_action(make_state(hand=[]), shop,
                                                   raw_g={"jokers": {"cards": []}}))


class TestPlayPathStaysClean(unittest.TestCase):
    def test_no_play_module_imports_the_probe_policies(self):
        """`stress` holds hand-written policies; importing one into play would
        make Python the player."""
        for name in PLAY_PATH_MODULES:
            path = ROOT / "sidecar" / name
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for mod in names:
                    for forbidden in ("stress", "qbench"):
                        self.assertNotIn(forbidden, mod, f"{name} imports {mod}")
                    self.assertNotIn("sim", mod.split(".")[-1:], f"{name} imports {mod}")

    def test_sim_does_not_choose_moves(self):
        """The simulator executes the action it is handed and nothing else."""
        source = (ROOT / "sidecar" / "sim.py").read_text()
        self.assertNotIn("find_best_playable_hands", source)
        self.assertNotIn("recommend_discards", source)


if __name__ == "__main__":
    unittest.main()


class TestConfidenceIsReportedNotObeyed(unittest.TestCase):
    """A near-coin-flip answer is still the model's answer."""

    def test_a_low_confidence_answer_is_carried_out(self):
        backend = ScriptedBackend({"which_aim": "pair_8", "send_filler_cards": False})
        backend.last_confidence = 0.02
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        action = agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        self.assertEqual({c.rank for c in action.cards}, {"8"})

    def test_low_confidence_decisions_are_recorded(self):
        backend = ScriptedBackend({"which_aim": "pair_8", "send_filler_cards": False})
        backend.last_confidence = 0.02
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        self.assertTrue(agent.low_confidence_decisions)
        self.assertIn("which_aim", [q for q, _ in agent.low_confidence_decisions])

    def test_a_confident_answer_is_not_flagged(self):
        backend = ScriptedBackend({"which_aim": "pair_8", "send_filler_cards": False})
        backend.last_confidence = 0.9
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        self.assertFalse(agent.low_confidence_decisions)
        self.assertEqual(agent.decision_confidence["which_aim"], 0.9)

    def test_backends_without_confidence_are_not_flagged(self):
        backend = ScriptedBackend({"which_aim": "pair_8", "send_filler_cards": False})
        agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
        agent.decide_hand_action(make_state(), raw_g={"hands": {}})
        self.assertFalse(agent.low_confidence_decisions)
