import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sidecar.client import BalatroBotClient
from sidecar.agent import BalatroGameState, AgentAction

console = Console()

class BalatroLayaRunner:
    """Autonomous HTTP JSON-RPC 2.0 Runner connecting Laya Agent to coder/balatrobot."""

    def __init__(self, endpoint: str = "http://127.0.0.1:12346", rules_path: str = "rules/balatro-strategy.md"):
        self.endpoint = endpoint
        self.client = BalatroBotClient(endpoint=endpoint, rules_path=rules_path)
        self.agent = self.client.agent
        self.last_state_name = None

    def handle_hand_decision(self, raw_g: dict):
        state = self.client.parse_state(raw_g)
        action: AgentAction = self.agent.decide_hand_action(state)

        table = Table(title=f"🃏 Hand in Play (Ante {state.ante} | {state.blind_name} | Score: {state.current_chips}/{state.target_chips})", show_header=True)
        table.add_column("Idx", justify="center", style="cyan")
        table.add_column("Rank", justify="center", style="magenta")
        table.add_column("Suit", justify="center", style="green")
        table.add_column("Selected", justify="center", style="bold yellow")

        selected_ids = {c.id for c in action.cards}
        for c in state.hand:
            sel_mark = "[bold yellow]YES[/bold yellow]" if c.id in selected_ids else "-"
            table.add_row(str(c.id), str(c.rank), str(c.suit), sel_mark)

        console.print(table)
        console.print(f"[bold cyan]Action:[/bold cyan] {action.action_type} | [bold cyan]Reasoning:[/bold cyan] {action.reasoning}")

        # In coder/balatrobot, card indices in play/discard are 0-based
        card_indices = [c.id for c in action.cards]
        if action.action_type == "PLAY":
            console.print(f"[bold green]>> PLAYING CARDS:[/bold green] {card_indices}")
            self.client.play_cards(card_indices)
        elif action.action_type == "DISCARD":
            console.print(f"[bold red]>> DISCARDING CARDS:[/bold red] {card_indices}")
            self.client.discard_cards(card_indices)

    def handle_shop_decision(self, raw_g: dict):
        shop_cards_raw = raw_g.get("shop", {}).get("cards", [])
        dollars = raw_g.get("money", 0)
        current_jokers = [j.get("label") or j.get("key", "") for j in raw_g.get("jokers", {}).get("cards", [])]
        max_jokers = raw_g.get("jokers", {}).get("max", 5)
        ante = raw_g.get("ante_num", 1)

        # Normalize shop items format for the agent
        shop_items = []
        for c in shop_cards_raw:
            shop_items.append({
                "name": c.get("label") or c.get("key", "Unknown"),
                "cost": c.get("cost", {}).get("buy", 4),
                "key": c.get("key", "")
            })

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

        if shop_items:
            table = Table(title=f"🛍️ Shop Inventory (${dollars} cash | Jokers: {len(current_jokers)}/{max_jokers})", show_header=True)
            table.add_column("Idx", justify="center", style="cyan")
            table.add_column("Item", justify="left", style="magenta")
            table.add_column("Cost", justify="center", style="green")
            for idx, it in enumerate(shop_items):
                table.add_row(str(idx), it["name"], f"${it['cost']}")
            console.print(table)

        decision = self.agent.decide_shop_action(state, shop_items, max_jokers=max_jokers)
        if decision is not None:
            one_based_idx, reasoning = decision
            zero_based_idx = one_based_idx - 1
            console.print(f"[bold green]🛒 Shop Decision:[/bold green] {reasoning}")
            console.print(f"[bold green]>> BUYING ITEM AT INDEX:[/bold green] {zero_based_idx}")
            self.client.buy_card(zero_based_idx)
            time.sleep(0.5)
        else:
            console.print(f"[bold yellow]🚪 Shop Decision:[/bold yellow] Ending shop (bankroll preserved for interest).")
            console.print(f"[bold yellow]>> LEAVING SHOP (NEXT ROUND)[/bold yellow]")
            self.client.next_round()
            time.sleep(0.5)

    def run_step(self) -> bool:
        raw_state = self.client.get_gamestate()
        if not raw_state:
            return False

        current_state = raw_state.get("state", "UNKNOWN")

        if current_state != self.last_state_name:
            console.print(f"[bold blue]⚡ Game State:[/bold blue] [white]{current_state}[/white]")
            self.last_state_name = current_state

        if current_state == "SELECTING_HAND":
            cards = raw_state.get("hand", {}).get("cards", [])
            if not cards:
                return False
            self.handle_hand_decision(raw_state)
            return True

        elif current_state == "BLIND_SELECT":
            console.print("[bold cyan]Blind Selection:[/bold cyan] Selecting Blind...")
            self.client.select_blind()
            return True

        elif current_state == "ROUND_EVAL":
            console.print("[bold green]Round Won:[/bold green] Cashing out rewards...")
            self.client.cash_out()
            return True

        elif current_state == "SHOP":
            self.handle_shop_decision(raw_state)
            return True

        elif current_state == "GAME_OVER":
            console.print(f"[bold red]💀 Run Ended at Ante {raw_state.get('ante_num', '?')}. Waiting...[/bold red]")
            time.sleep(2.0)
            return False

        elif current_state == "MENU":
            # Auto-start on Profile 3 if in menu
            console.print("[dim]In Menu - Starting new Red Deck White Stake run...[/dim]")
            self.client.start_game("RED", "WHITE")
            time.sleep(1.0)
            return True

        return False

    def run_loop(self):
        console.print(Panel.fit(
            "[bold green]Balatro Laya Autonomous Bridge Active[/bold green]\n"
            f"API Endpoint: [cyan]{self.endpoint}[/cyan] (HTTP JSON-RPC 2.0)\n"
            "Mod: [green]coder/balatrobot v1.5.2[/green]\n"
            "Engine: [magenta]Laya ModernBERT-large + Combinatorial Math[/magenta]",
            title="Laya Balatro Agent"
        ))
        while True:
            try:
                acted = self.run_step()
                time.sleep(0.2 if acted else 0.8)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Agent stopped by user.[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error in loop: {e}[/red]")
                time.sleep(1.0)

if __name__ == "__main__":
    runner = BalatroLayaRunner()
    runner.run_loop()
