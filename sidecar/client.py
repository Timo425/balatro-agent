import time
import httpx
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel

from sidecar.poker_eval import Card
from sidecar.agent import BalatroAgent, BalatroGameState, AgentAction

console = Console()

class BalatroBotClient:
    """HTTP JSON-RPC 2.0 client communicating with BalatroBot Lua mod."""

    def __init__(self, endpoint: str = "http://127.0.0.1:12345", rules_path: str = "rules/balatro-strategy.md"):
        self.endpoint = endpoint
        self.agent = BalatroAgent(rules_path=rules_path)
        self.rpc_id = 1

    def send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.rpc_id
        }
        self.rpc_id += 1
        try:
            resp = httpx.post(self.endpoint, json=payload, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result")
        except httpx.ConnectError:
            return None
        except Exception as e:
            console.print(f"[red]RPC Error ({method}): {e}[/red]")
            return None

    def get_state(self) -> Optional[Dict[str, Any]]:
        return self.send_rpc("get_gamestate")

    def parse_state(self, raw_g: Dict[str, Any]) -> BalatroGameState:
        """Translates BalatroBot JSON-RPC state to BalatroGameState."""
        raw_hand = raw_g.get("hand", [])
        hand_cards = []
        for i, c in enumerate(raw_hand):
            # BalatroBot format: suit string, rank string
            suit = c.get("suit", "Spades")
            rank = str(c.get("value", "2"))
            hand_cards.append(Card(rank=rank, suit=suit, id=i))

        jokers = [j.get("name", "Joker") for j in raw_g.get("jokers", [])]

        return BalatroGameState(
            ante=raw_g.get("ante", 1),
            blind_name=raw_g.get("blind", {}).get("name", "Small Blind"),
            target_chips=raw_g.get("blind", {}).get("chips", 300),
            current_chips=raw_g.get("chips", 0),
            hands_left=raw_g.get("hands_left", 4),
            discards_left=raw_g.get("discards_left", 3),
            money=raw_g.get("dollars", 4),
            hand=hand_cards,
            jokers=jokers
        )

    def play_round_step(self, raw_g: Dict[str, Any]) -> bool:
        """Executes a single game decision using the agent."""
        state = self.parse_state(raw_g)
        action: AgentAction = self.agent.decide_hand_action(state)

        console.print(f"[bold cyan]Agent Decision:[/bold cyan] [yellow]{action.action_type}[/yellow] ({action.reasoning})")

        card_indices = [c.id for c in action.cards]

        if action.action_type == "PLAY":
            res = self.send_rpc("play_hand", {"indices": card_indices})
            return res is not None
        elif action.action_type == "DISCARD":
            res = self.send_rpc("discard_hand", {"indices": card_indices})
            return res is not None

        return False

    def run_loop(self, poll_interval: float = 1.0):
        """Main autonomous bot loop."""
        console.print(Panel.fit("[bold green]BalatroBot Agent Started[/bold green]\nPolling game on " + self.endpoint))
        while True:
            raw_state = self.get_state()
            if not raw_state:
                console.print("[dim]Waiting for Balatro to be running on " + self.endpoint + "...[/dim]")
            else:
                phase = raw_state.get("state", "UNKNOWN")
                if phase == "ROUND":
                    self.play_round_step(raw_state)
                elif phase == "BLIND_SELECT":
                    console.print("[green]Selecting blind...[/green]")
                    self.send_rpc("select_blind")
                elif phase == "SHOP":
                    console.print("[yellow]In shop. Finishing shop phase...[/yellow]")
                    self.send_rpc("leave_shop")

            time.sleep(poll_interval)

if __name__ == "__main__":
    client = BalatroBotClient()
    client.run_loop()
