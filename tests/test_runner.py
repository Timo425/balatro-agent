"""Runner integration: each game state drives the right client call.

No game is needed -- the JSON-RPC client is replaced with a recorder. This is the
test that catches a decision signature drifting out of step with the runner.
"""

import unittest
from typing import Any, Dict, List, Optional

from sidecar.agent import BalatroAgent
from sidecar.runner import BalatroLayaRunner
from tests.support import ScriptedBackend


class FakeClient:
    def __init__(self, gamestate: Dict[str, Any], agent: Any):
        self.gamestate = gamestate
        self.agent = agent
        self.calls: List[str] = []

    def _record(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            return {"ok": True}
        return call

    def __getattr__(self, name):
        return self._record(name)

    def get_gamestate(self):
        self.calls.append("gamestate")
        return self.gamestate

    def parse_state(self, raw_g):
        from sidecar.client import BalatroBotClient
        return BalatroBotClient.parse_state(self, raw_g)


def card(rank, suit):
    return {"value": {"rank": rank, "suit": suit}, "state": {}, "modifier": {}}


def runner_for(gamestate, answers):
    backend = ScriptedBackend(answers)
    agent = BalatroAgent(backend=backend, tree_mode=True, poll_mode=False)
    runner = BalatroLayaRunner(agent=agent)
    runner.client = FakeClient(gamestate, agent)
    return runner, backend


HAND_STATE = {
    "state": "SELECTING_HAND",
    "ante_num": 1,
    "money": 4,
    "hands": {},
    "jokers": {"cards": [], "count": 0, "max": 5},
    "consumables": {"cards": []},
    "hand": {"cards": [card("A", "Spades"), card("K", "Spades"), card("Q", "Spades"),
                       card("J", "Spades"), card("9", "Spades"), card("2", "Hearts"),
                       card("3", "Clubs"), card("4", "Diamonds")]},
    "blinds": {"small": {"status": "CURRENT", "name": "Small Blind", "chips": 300}},
    "round": {"chips": 0, "hands_left": 4, "discards_left": 3},
}


class TestRunnerSteps(unittest.TestCase):
    def test_selecting_hand_plays_what_the_model_chose(self):
        runner, backend = runner_for(HAND_STATE, {
            "dig_for_better_hand": False,
            "which_hand": lambda c: next(k for k in c if k.startswith("flush_")),
            "send_filler_cards": False,
        })
        self.assertTrue(runner.run_step())
        self.assertIn("play_cards", runner.client.calls)
        self.assertNotIn("discard_cards", runner.client.calls)

    def test_selecting_hand_discards_when_the_model_says_dig(self):
        state = dict(HAND_STATE)
        state["hand"] = {"cards": [card("K", "Hearts"), card("10", "Hearts"),
                                   card("7", "Hearts"), card("4", "Hearts"),
                                   card("2", "Spades"), card("Q", "Clubs"),
                                   card("9", "Diamonds"), card("3", "Spades")]}
        runner, _ = runner_for(state, {"dig_for_better_hand": True,
                                       "which_discard": "flush_draw_hearts"})
        self.assertTrue(runner.run_step())
        self.assertIn("discard_cards", runner.client.calls)

    def test_blind_select_plays_or_skips_as_answered(self):
        state = {"state": "BLIND_SELECT", "ante_num": 1, "money": 4,
                 "jokers": {"cards": []}, "consumables": {"cards": []},
                 "blinds": {"small": {"status": "SELECT", "name": "Small Blind",
                                      "chips": 300, "tag_name": "Rare Tag"}},
                 "round": {"chips": 0, "hands_left": 4, "discards_left": 3}}
        runner, _ = runner_for(state, {"skip_blind": True})
        runner.run_step()
        self.assertIn("skip_blind", runner.client.calls)

        runner, _ = runner_for(state, {"skip_blind": False})
        runner.run_step()
        self.assertIn("select_blind", runner.client.calls)

    def test_shop_buys_the_chosen_item(self):
        state = {"state": "SHOP", "ante_num": 1, "money": 12,
                 "jokers": {"cards": [], "count": 0, "max": 5},
                 "consumables": {"cards": []},
                 "shop": {"cards": [
                     {"label": "Joker", "key": "j_joker", "set": "JOKER",
                      "cost": {"buy": 4, "sell": 2}, "value": {"effect": "+4 Mult"}},
                     {"label": "Banner", "key": "j_banner", "set": "JOKER",
                      "cost": {"buy": 5, "sell": 2},
                      "value": {"effect": "+30 Chips for each remaining discard"}}]},
                 "vouchers": {"cards": []}, "packs": {"cards": []},
                 "round": {"chips": 0, "hands_left": 4, "discards_left": 3}}
        runner, _ = runner_for(state, {"shop_category": "cat_joker",
                                       "which_shop_item": "buy1_banner"})
        runner.run_step()
        self.assertIn("buy_card", runner.client.calls)

    def test_shop_leaves_when_the_model_buys_nothing(self):
        state = {"state": "SHOP", "ante_num": 1, "money": 12,
                 "jokers": {"cards": [], "count": 0, "max": 5},
                 "consumables": {"cards": []},
                 "shop": {"cards": [{"label": "Joker", "key": "j_joker", "set": "JOKER",
                                     "cost": {"buy": 4, "sell": 2}}]},
                 "vouchers": {"cards": []},
                 "packs": {"cards": [{"label": "Arcana Pack", "key": "p_arcana",
                                      "cost": {"buy": 4}}]},
                 "round": {"chips": 0, "hands_left": 4, "discards_left": 3}}
        runner, _ = runner_for(state, {"shop_category": "cat_save"})
        runner.run_step()
        self.assertIn("next_round", runner.client.calls)
        self.assertNotIn("buy_pack", runner.client.calls,
                         "Python must not buy a pack the model did not choose")

    def test_booster_pack_takes_the_chosen_card(self):
        state = {"state": "SMODS_BOOSTER_OPENED", "ante_num": 1, "money": 4,
                 "jokers": {"cards": [], "count": 0, "max": 5},
                 "consumables": {"cards": []},
                 "pack": {"cards": [{"label": "Mercury", "key": "c_mercury", "set": "PLANET"},
                                    {"label": "Jupiter", "key": "c_jupiter", "set": "PLANET"}]},
                 "hand": {"cards": []},
                 "blinds": {}, "round": {"chips": 0, "hands_left": 4, "discards_left": 3}}
        runner, _ = runner_for(state, {"pack_choice": "take1_jupiter"})
        runner.run_step()
        self.assertIn("select_pack_card", runner.client.calls)

    def test_round_eval_cashes_out(self):
        state = {"state": "ROUND_EVAL", "ante_num": 1, "money": 4,
                 "round": {"chips": 300, "hands_left": 2, "discards_left": 2}}
        runner, _ = runner_for(state, {})
        runner.run_step()
        self.assertIn("cash_out", runner.client.calls)

    def test_fallbacks_are_announced_once_each(self):
        runner, backend = runner_for(HAND_STATE, {"dig_for_better_hand": False})
        runner.run_step()
        self.assertEqual(runner._reported_fallbacks, backend.fallbacks)


if __name__ == "__main__":
    unittest.main()
