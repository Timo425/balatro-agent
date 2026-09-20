import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import httpx
from typing import Dict, Any, Optional, List, Tuple
from rich.console import Console
from rich.panel import Panel

from sidecar.poker_eval import Card
from sidecar.agent import BalatroAgent, BalatroGameState, AgentAction

console = Console()

SUIT_MAP = {
    "H": "Hearts",
    "D": "Diamonds",
    "C": "Clubs",
    "S": "Spades",
    "Hearts": "Hearts",
    "Diamonds": "Diamonds",
    "Clubs": "Clubs",
    "Spades": "Spades",
}

RANK_MAP = {
    "T": "10",
    "Ace": "A",
    "King": "K",
    "Queen": "Q",
    "Jack": "J",
}

class BalatroBotClient:
    """HTTP JSON-RPC 2.0 client communicating with coder/balatrobot v1.5.2+."""

    def __init__(self, endpoint: str = "http://127.0.0.1:12346", rules_path: str = "rules/balatro-strategy.md"):
        self.endpoint = endpoint
        self.agent = BalatroAgent(rules_path=rules_path)
        self.rpc_id = 1
        self.client = httpx.Client(base_url=self.endpoint, timeout=30.0)

    def send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.rpc_id
        }
        self.rpc_id += 1
        try:
            resp = self.client.post("/", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    err = data["error"]
                    console.print(f"[yellow]RPC Notice ({method}): {err.get('message', err)}[/yellow]")
                    return None
                return data.get("result")
            else:
                console.print(f"[red]HTTP {resp.status_code} ({method}): {resp.text}[/red]")
                return None
        except httpx.ConnectError:
            return None
        except Exception as e:
            console.print(f"[red]RPC Error ({method}): {e}[/red]")
            return None

    def health(self) -> bool:
        res = self.send_rpc("health")
        return res is not None and res.get("status") == "ok"

    def get_gamestate(self) -> Optional[Dict[str, Any]]:
        return self.send_rpc("gamestate")

    def select_blind(self) -> Optional[Dict[str, Any]]:
        return self.send_rpc("select")

    def play_cards(self, card_indices: List[int]) -> Optional[Dict[str, Any]]:
        return self.send_rpc("play", {"cards": card_indices})

    def discard_cards(self, card_indices: List[int]) -> Optional[Dict[str, Any]]:
        return self.send_rpc("discard", {"cards": card_indices})

    def cash_out(self) -> Optional[Dict[str, Any]]:
        return self.send_rpc("cash_out")

    def buy_card(self, index: int) -> Optional[Dict[str, Any]]:
        return self.send_rpc("buy", {"card": index})

    def buy_pack(self, index: int) -> Optional[Dict[str, Any]]:
        return self.send_rpc("buy", {"pack": index})

    def buy_voucher(self, index: int) -> Optional[Dict[str, Any]]:
        return self.send_rpc("buy", {"voucher": index})

    def next_round(self) -> Optional[Dict[str, Any]]:
        return self.send_rpc("next_round")

    def rearrange_jokers(self, order: List[int]) -> Optional[Dict[str, Any]]:
        return self.send_rpc("rearrange", {"jokers": order})

    def use_consumable(self, consumable_idx: int, target_cards: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {"consumable": consumable_idx}
        if target_cards:
            params["cards"] = target_cards
        return self.send_rpc("use", params)

    def select_pack_card(self, card_idx: Optional[int] = None, targets: Optional[List[int]] = None, skip: bool = False) -> Optional[Dict[str, Any]]:
        if skip:
            return self.send_rpc("pack", {"skip": True})
        params: Dict[str, Any] = {"card": card_idx if card_idx is not None else 0}
        if targets:
            params["targets"] = targets
        return self.send_rpc("pack", params)

    def start_game(self, deck: str = "RED", stake: str = "WHITE") -> Optional[Dict[str, Any]]:
        return self.send_rpc("start", {"deck": deck, "stake": stake})

    def parse_state(self, raw_g: Dict[str, Any]) -> BalatroGameState:
        """Translates coder/balatrobot gamestate to BalatroGameState."""
        raw_hand = raw_g.get("hand", {}).get("cards", [])
        hand_cards = []
        for i, c in enumerate(raw_hand):
            val_obj = c.get("value", {})
            raw_suit = val_obj.get("suit", "S")
            raw_rank = val_obj.get("rank", "2")

            suit = SUIT_MAP.get(raw_suit, raw_suit)
            rank = RANK_MAP.get(raw_rank, raw_rank)
            hand_cards.append(Card(rank=rank, suit=suit, id=i))

        raw_jokers = raw_g.get("jokers", {}).get("cards", [])
        jokers = [j.get("label") or j.get("key", "Joker") for j in raw_jokers]

        blinds = raw_g.get("blinds", {})
        target_chips = 300
        blind_name = "Small Blind"
        for b_key, b_info in blinds.items():
            status = b_info.get("status", "")
            if status in ("SELECTING", "CURRENT", "ACTIVE") or b_info.get("selected"):
                target_chips = b_info.get("chips", 300)
                blind_name = b_info.get("name", b_key.capitalize() + " Blind")
                break

        round_info = raw_g.get("round", {})
        current_chips = round_info.get("chips", 0)
        hands_left = round_info.get("hands_left", 4)
        discards_left = round_info.get("discards_left", 3)

        return BalatroGameState(
            ante=raw_g.get("ante_num", 1),
            blind_name=blind_name,
            target_chips=target_chips,
            current_chips=current_chips,
            hands_left=hands_left,
            discards_left=discards_left,
            money=raw_g.get("money", 4),
            hand=hand_cards,
            jokers=jokers
        )
