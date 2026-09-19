import sys
import json
import time
import socket
from pathlib import Path
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
        self.last_state_signature = ""

    def send_cmd(self, cmd: str):
        console.print(f"[bold green]>> SENDING COMMAND:[/bold green] [bold yellow]{cmd}[/bold yellow]")
        self.sock.sendto(cmd.encode("utf-8"), self.addr)

    def parse_gamestate(self, raw_g: dict) -> BalatroGameState:
        hand_cards = []
        for i, c in enumerate(raw_g.get("hand", []), start=1):
            val_str = str(c.get("value", "2"))
            suit_str = str(c.get("suit", "Spades"))
            hand_cards.append(Card(rank=val_str, suit=suit_str, id=i))

        jokers = [j.get("name", "Joker") for j in raw_g.get("jokers", [])]

        blind_data = raw_g.get("blind", {}) or {}
        round_data = raw_g.get("current_round", {}) or {}

        return BalatroGameState(
            ante=raw_g.get("round", 1),
            blind_name=blind_data.get("name", "Blind"),
            target_chips=blind_data.get("chips", 300),
            current_chips=raw_g.get("dollars", 0), # Chips scored this round
            hands_left=round_data.get("hands_left", 4),
            discards_left=round_data.get("discards_left", 3),
            money=raw_g.get("dollars", 4),
            hand=hand_cards,
            jokers=jokers
        )

    def handle_hand_decision(self, raw_g: dict):
        state = self.parse_gamestate(raw_g)
        action: AgentAction = self.agent.decide_hand_action(state)

        # Build clean visual table of hand
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
        elif action.action_type == "DISCARD":
            self.send_cmd(f"DISCARD_HAND|{idx_str}")

    def run_loop(self):
        console.print(Panel.fit(
            "[bold green]Balatro Laya Autonomous Bridge Active[/bold green]\n"
            f"Connected to: [cyan]{self.addr[0]}:{self.addr[1]}[/cyan]\n"
            "Engine: [magenta]Laya ModernBERT-large + Combinatorial Math[/magenta]",
            title="Laya Balatro Agent"
        ))

        while True:
            try:
                self.sock.sendto(b"HELLO", self.addr)
                data, _ = self.sock.recvfrom(65535)
                g = json.loads(data.decode("utf-8", errors="replace"))

                waiting_for = g.get("waitingFor")
                is_action_ready = g.get("waitingForAction", False)

                if is_action_ready:
                    if waiting_for == "select_cards_from_hand":
                        self.handle_hand_decision(g)
                        time.sleep(1.0)
                    elif waiting_for == "skip_or_select_blind":
                        console.print("[bold cyan]Blind Selection:[/bold cyan] Selecting Blind...")
                        self.send_cmd("SELECT_BLIND")
                        time.sleep(1.0)
                    elif waiting_for == "select_shop_action":
                        console.print("[bold yellow]Shop Phase:[/bold yellow] Ending shop...")
                        self.send_cmd("END_SHOP")
                        time.sleep(1.0)
                    elif waiting_for == "start_run":
                        console.print("[dim]Waiting for game run to start... (Click 'Play' -> 'New Run' in Balatro)[/dim]")
                        time.sleep(1.0)

            except socket.timeout:
                console.print("[dim]Waiting for Balatro on UDP 12345...[/dim]")
                time.sleep(1.0)
            except Exception as e:
                console.print(f"[red]Loop error: {e}[/red]")
                time.sleep(1.0)

if __name__ == "__main__":
    runner = BalatroLayaRunner()
    runner.run_loop()
