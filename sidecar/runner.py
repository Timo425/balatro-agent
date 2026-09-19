import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import socket
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sidecar.poker_eval import Card, PokerEvaluator
from sidecar.agent import BalatroAgent, BalatroGameState, AgentAction

console = Console()

class BalatroLayaRunner:
    """Autonomous UDP Bridge between Balatrobot mod and Laya decision engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12345, rules_path: str = "rules/balatro-strategy.md"):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.5)
        self.agent = BalatroAgent(rules_path=rules_path)

    def send_cmd(self, cmd: str):
        console.print(f"[bold green]>> SENDING COMMAND:[/bold green] [bold yellow]{cmd}[/bold yellow]")
        self.sock.sendto(cmd.encode("utf-8"), self.addr)

    def parse_gamestate(self, raw_g: dict) -> BalatroGameState:
        hand_cards = []
        val_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
        for i, c in enumerate(raw_g.get("hand", []), start=1):
            val_raw = str(c.get("value", "2"))
            val_clean = val_map.get(val_raw, val_raw)
            suit_str = str(c.get("suit", "Spades"))
            hand_cards.append(Card(rank=val_clean, suit=suit_str, id=i))

        jokers = [j.get("name", "Joker") for j in raw_g.get("jokers", [])]
        blind_data = raw_g.get("blind", {}) or {}
        round_data = raw_g.get("current_round", {}) or {}

        return BalatroGameState(
            ante=raw_g.get("round", 1),
            blind_name=blind_data.get("name", "Small Blind"),
            target_chips=blind_data.get("chips", 300),
            current_chips=raw_g.get("chips", 0),
            hands_left=round_data.get("hands_left", 4),
            discards_left=round_data.get("discards_left", 3),
            money=raw_g.get("dollars", 4),
            hand=hand_cards,
            jokers=jokers
        )

    def handle_hand_decision(self, raw_g: dict):
        state = self.parse_gamestate(raw_g)
        action: AgentAction = self.agent.decide_hand_action(state)

        table = Table(title=f"Hand in Play (Ante {state.ante} - {state.blind_name})", show_header=True)
        table.add_column("Idx", justify="center", style="cyan")
        table.add_column("Rank", justify="center", style="magenta")
        table.add_column("Suit", justify="center", style="green")
        table.add_column("Selected", justify="center", style="bold yellow")

        selected_ids = {c.id for c in action.cards}
        for c in state.hand:
            sel_mark = "[bold yellow]YES[/bold yellow]" if c.id in selected_ids else "-"
            table.add_row(str(c.id), c.rank, c.suit, sel_mark)

        console.print(table)
        console.print(f"[bold cyan]Action:[/bold cyan] {action.action_type} | [bold cyan]Reasoning:[/bold cyan] {action.reasoning}")

        idx_str = ",".join(str(c.id) for c in action.cards)
        if action.action_type == "PLAY":
            self.send_cmd(f"PLAY_HAND|{idx_str}")
            time.sleep(0.3)
        elif action.action_type == "DISCARD":
            self.send_cmd(f"DISCARD_HAND|{idx_str}")
            time.sleep(0.3)

    def handle_shop_decision(self, raw_g: dict):
        shop = raw_g.get("shop", {})
        shop_cards = shop.get("cards", [])
        dollars = raw_g.get("dollars", 0)
        current_jokers = [j.get("name", "") for j in raw_g.get("jokers", [])]
        max_jokers = raw_g.get("max_jokers", 5)

        round_data = raw_g.get("round_resets", {}) or {}
        ante = round_data.get("ante", raw_g.get("round", 1))

        state = BalatroGameState(
            ante=ante,
            blind_name="Shop",
            target_chips=0,
            current_chips=0,
            hands_left=0,
            discards_left=0,
            money=dollars,
            hand=[],
            jokers=current_jokers
        )

        decision = self.agent.decide_shop_action(state, shop_cards, max_jokers=max_jokers)
        if decision is not None:
            buy_idx, reasoning = decision
            console.print(f"[bold green]🛒 Shop Action:[/bold green] {reasoning}")
            self.send_cmd(f"BUY_CARD|{buy_idx}")
            time.sleep(0.4)
        else:
            console.print(f"[bold yellow]🚪 Shop Phase:[/bold yellow] Ending shop ( bankroll saved).")
            self.send_cmd("END_SHOP")
            time.sleep(0.3)

    def run_step(self) -> bool:
        try:
            self.sock.sendto(b"HELLO", self.addr)
            data, _ = self.sock.recvfrom(65535)
            g = json.loads(data.decode("utf-8", errors="replace"))

            waiting_for = g.get("waitingFor")
            is_action_ready = g.get("waitingForAction", False)

            if not is_action_ready:
                return False

            if waiting_for == "select_cards_from_hand":
                if not g.get("hand"):
                    return False
                self.handle_hand_decision(g)
                return True
            elif waiting_for == "skip_or_select_blind":
                console.print("[bold cyan]Blind Selection:[/bold cyan] Selecting Blind...")
                self.send_cmd("SELECT_BLIND")
                time.sleep(0.4)
                return True
            elif waiting_for == "sell_jokers":
                self.send_cmd("SELL_JOKER|")
                return True
            elif waiting_for == "rearrange_jokers":
                self.send_cmd("REARRANGE_JOKERS|")
                return True
            elif waiting_for == "use_or_sell_consumables":
                self.send_cmd("USE_CONSUMABLE|")
                return True
            elif waiting_for == "rearrange_consumables":
                self.send_cmd("REARRANGE_CONSUMABLES|")
                return True
            elif waiting_for == "rearrange_hand":
                self.send_cmd("REARRANGE_HAND|")
                return True
            elif waiting_for == "select_shop_action":
                self.handle_shop_decision(g)
                return True

        except socket.timeout:
            return False
        except Exception as e:
            console.print(f"[red]Error in step: {e}[/red]")
            return False

        return False

    def run_loop(self):
        console.print(Panel.fit(
            "[bold green]Balatro Laya Autonomous Bridge Active[/bold green]\n"
            f"Connected to: [cyan]{self.addr[0]}:{self.addr[1]}[/cyan]\n"
            "Engine: [magenta]Laya ModernBERT-large + Combinatorial Math[/magenta]",
            title="Laya Balatro Agent"
        ))
        while True:
            acted = self.run_step()
            time.sleep(0.05 if acted else 0.1)

if __name__ == "__main__":
    runner = BalatroLayaRunner()
    runner.run_loop()
