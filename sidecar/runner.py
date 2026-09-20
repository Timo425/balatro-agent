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

from sidecar.client import BalatroBotClient, extract_cards_list
from sidecar.agent import BalatroAgent, BalatroGameState, AgentAction

console = Console()

class BalatroLayaRunner:
    """Autonomous HTTP JSON-RPC 2.0 Runner connecting Laya Agent to coder/balatrobot."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:12346",
        rules_path: str = "rules/balatro-strategy.md",
        agent: Optional[Any] = None,
        backend: str = "laya",
        model: Optional[str] = None,
        **backend_kwargs
    ):
        self.endpoint = endpoint
        if agent is None:
            agent = BalatroAgent(rules_path=rules_path, backend=backend, model=model, **backend_kwargs)
        self.client = BalatroBotClient(endpoint=endpoint, rules_path=rules_path, agent=agent)
        self.agent = self.client.agent
        self.last_state_name = None

    def check_joker_rearrange(self, raw_g: dict):
        """Ensures RULE_JOKER_ORDER (+Chips -> +Flat Mult -> +Scaling -> X-Mult)."""
        jokers = extract_cards_list(raw_g, "jokers")
        if len(jokers) > 1:
            new_order = self.agent.decide_joker_order(jokers)
            if new_order is not None:
                console.print(f"[bold yellow]🔄 RULE_JOKER_ORDER:[/bold yellow] Reordering Jokers to {new_order} (X-Mult to right)")
                self.client.rearrange_jokers(new_order)
                time.sleep(0.3)

    def check_consumables(self, raw_g: dict) -> bool:
        """Ensures RULE_USE_PLANET & RULE_USE_TAROT."""
        consumables = extract_cards_list(raw_g, "consumables")
        if not consumables:
            return False

        hand_cards = self.client.parse_state(raw_g).hand if raw_g.get("hand") else []
        decision = self.agent.decide_consumable_action(consumables, hand=hand_cards, raw_g=raw_g)
        if decision is not None:
            idx, targets, reasoning = decision
            console.print(f"[bold magenta]✨ Consumable Action:[/bold magenta] {reasoning}")
            res = self.client.use_consumable(idx, target_cards=targets)
            time.sleep(0.4)
            if res is not None and not (isinstance(res, dict) and 'error' in res):
                return True
            console.print(f"[yellow]Consumable use ei onnestunud: {res}[/yellow]")
            return False
        return False

    def handle_hand_decision(self, raw_g: dict):
        self.check_joker_rearrange(raw_g)

        if self.check_consumables(raw_g):
            return

        state = self.client.parse_state(raw_g)
        action: AgentAction = self.agent.decide_hand_action(state, raw_g=raw_g)

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

        card_indices = [c.id for c in action.cards]
        if action.action_type == "PLAY":
            console.print(f"[bold green]>> PLAYING CARDS:[/bold green] {card_indices}")
            self.client.play_cards(card_indices)
        elif action.action_type == "DISCARD":
            console.print(f"[bold red]>> DISCARDING CARDS:[/bold red] {card_indices}")
            self.client.discard_cards(card_indices)

    def handle_shop_decision(self, raw_g: dict):
        if self.check_consumables(raw_g):
            return

        self.check_joker_rearrange(raw_g)

        shop_cards_raw = extract_cards_list(raw_g, "shop")
        vouchers_raw = extract_cards_list(raw_g, "vouchers")
        packs_raw = extract_cards_list(raw_g, "packs")
        dollars = raw_g.get("money", 0)
        current_jokers = [j.get("label") or j.get("key", "") for j in extract_cards_list(raw_g, "jokers") if isinstance(j, dict)]
        j_obj = raw_g.get("jokers")
        max_jokers = j_obj.get("max", 5) if isinstance(j_obj, dict) else 5
        ante = raw_g.get("ante_num", 1)

        shop_items = []
        for c in shop_cards_raw:
            if not isinstance(c, dict):
                continue
            cost_val = c.get("cost")
            buy_cost = cost_val.get("buy", 4) if isinstance(cost_val, dict) else (cost_val if isinstance(cost_val, int) else 4)
            shop_items.append({
                "name": c.get("label") or c.get("key", "Unknown"),
                "cost": buy_cost,
                "key": c.get("key", "")
            })

        for v_idx, v in enumerate(vouchers_raw):
            if not isinstance(v, dict):
                continue
            cost_val = v.get("cost")
            buy_cost = cost_val.get("buy", 10) if isinstance(cost_val, dict) else (cost_val if isinstance(cost_val, int) else 10)
            shop_items.append({
                "name": f"Voucher: {v.get('label') or v.get('key', 'Unknown')}",
                "cost": buy_cost,
                "key": v.get("key", ""),
                "is_voucher": True,
                "voucher_idx": v_idx,
            })

        for p_idx, p in enumerate(packs_raw):
            if not isinstance(p, dict):
                continue
            cost_val = p.get("cost")
            buy_cost = cost_val.get("buy", 6) if isinstance(cost_val, dict) else (cost_val if isinstance(cost_val, int) else 6)
            shop_items.append({
                "name": f"Booster: {p.get('label') or p.get('key', 'Pack')}",
                "cost": buy_cost,
                "key": p.get("key", ""),
                "is_pack": True,
                "pack_idx": p_idx,
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

        decision = self.agent.decide_shop_action(state, shop_items, max_jokers=max_jokers, raw_g=raw_g)
        if decision is not None:
            act_type, target_idx, reasoning = decision
            if act_type == "SELL":
                console.print(f"[bold yellow]💰 Selling Joker:[/bold yellow] {reasoning}")
                self.client.sell_joker(target_idx)
                time.sleep(0.5)
                return
            elif act_type == "BUY":
                console.print(f"[bold green]🛒 Shop Decision:[/bold green] {reasoning}")
                console.print(f"[bold green]>> BUYING ITEM AT INDEX:[/bold green] {target_idx}")
                if target_idx < len(shop_items):
                    item_meta = shop_items[target_idx]
                    if item_meta.get("is_voucher"):
                        v_idx = item_meta["voucher_idx"]
                        console.print(f"[bold green]>> BUYING VOUCHER AT INDEX:[/bold green] {v_idx}")
                        self.client.buy_voucher(v_idx)
                    elif item_meta.get("is_pack"):
                        p_idx = item_meta["pack_idx"]
                        console.print(f"[bold green]>> BUYING BOOSTER PACK AT INDEX:[/bold green] {p_idx}")
                        self.client.buy_pack(p_idx)
                    else:
                        self.client.buy_card(target_idx)
                time.sleep(0.5)
                return

        # Check Booster packs
        packs_raw = extract_cards_list(raw_g, "packs")
        if packs_raw and (dollars >= 25 or (ante <= 2 and len(current_jokers) < 2)):
            for p_idx, pack in enumerate(packs_raw):
                if not isinstance(pack, dict):
                    continue
                p_cost = pack.get("cost")
                cost = p_cost.get("buy", 6) if isinstance(p_cost, dict) else (p_cost if isinstance(p_cost, int) else 6)
                if dollars - cost >= (20 if ante >= 3 else 0):
                    pack_name = pack.get("label") or pack.get("key", "Booster Pack")
                    console.print(f"[bold green]🎁 Buying Booster Pack:[/bold green] '{pack_name}' (${cost})")
                    self.client.buy_pack(p_idx)
                    time.sleep(0.5)
                    return

        console.print(f"[bold yellow]🚪 Shop Decision:[/bold yellow] Ending shop (bankroll preserved for interest).")
        console.print(f"[bold yellow]>> LEAVING SHOP (NEXT ROUND)[/bold yellow]")
        self.client.next_round()
        time.sleep(0.5)

    def handle_booster_pack(self, raw_g: dict):
        """Handles opened booster pack (SMODS_BOOSTER_OPENED state)."""
        pack_cards = extract_cards_list(raw_g, "pack")
        if not pack_cards:
            console.print("[yellow]Pack empty, skipping...[/yellow]")
            self.client.select_pack_card(skip=True)
            return

        state = self.client.parse_state(raw_g)
        j_obj = raw_g.get("jokers")
        max_jokers = j_obj.get("max", 5) if isinstance(j_obj, dict) else 5

        table = Table(title="🎁 Booster Pack Contents", show_header=True)
        table.add_column("Idx", justify="center", style="cyan")
        table.add_column("Card", justify="left", style="magenta")
        table.add_column("Type", justify="center", style="green")
        for idx, c in enumerate(pack_cards):
            table.add_row(str(idx), c.get("label") or c.get("key", ""), c.get("set", ""))
        console.print(table)

        decision = self.agent.decide_pack_card(
            pack_cards, 
            hand=state.hand, 
            jokers_count=len(state.jokers), 
            max_jokers=max_jokers
        )

        if decision is not None:
            card_idx, targets, reasoning = decision
            console.print(f"[bold green]🎯 Pack Decision:[/bold green] {reasoning} (targets={targets})")
            res = self.client.select_pack_card(card_idx=card_idx, targets=targets)
            if res is None:
                # If selection failed (e.g. invalid target count in game state), fallback to skip to avoid infinite loop
                console.print("[yellow]⚠️ Pack selection returned error, skipping pack to continue game...[/yellow]")
                self.client.select_pack_card(skip=True)
        else:
            console.print("[yellow]No suitable card or targets, skipping pack...[/yellow]")
            self.client.select_pack_card(skip=True)
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
            cards = extract_cards_list(raw_state, "hand")
            if not cards:
                return False
            self.handle_hand_decision(raw_state)
            return True

        elif current_state == "BLIND_SELECT":
            state = self.client.parse_state(raw_state)
            action, reasoning = self.agent.decide_blind_action(state, raw_g=raw_state)
            if action == "SKIP":
                console.print(f"[bold yellow]⏩ Blind Skip Decision:[/bold yellow] {reasoning}")
                res = self.client.skip_blind()
                if res is None:
                    console.print("[yellow]Skip unavailable, selecting blind instead...[/yellow]")
                    self.client.select_blind()
            else:
                console.print(f"[bold cyan]Blind Selection:[/bold cyan] {reasoning}")
                self.client.select_blind()
            time.sleep(0.5)
            return True

        elif current_state == "ROUND_EVAL":
            console.print("[bold green]Round Won:[/bold green] Cashing out rewards...")
            self.client.cash_out()
            return True

        elif current_state == "SHOP":
            self.handle_shop_decision(raw_state)
            return True

        elif current_state in ("SMODS_BOOSTER_OPENED", "BOOSTER_PACK"):
            self.handle_booster_pack(raw_state)
            return True

        elif current_state == "GAME_OVER":
            console.print(f"[bold red]💀 Run Ended at Ante {raw_state.get('ante_num', '?')}. Waiting...[/bold red]")
            time.sleep(2.0)
            return False

        elif current_state == "MENU":
            time.sleep(1.0)
            return False

        return False

    def run_loop(self):
        console.print(Panel.fit(
            "[bold green]Balatro Laya Autonomous Bridge Active[/bold green]\n"
            f"API Endpoint: [cyan]{self.endpoint}[/cyan] (HTTP JSON-RPC 2.0)\n"
            "Mod: [green]coder/balatrobot v1.5.2[/green]\n"
            "Engine: [magenta]Laya ModernBERT-large + Living Strategy Rulebook[/magenta]",
            title="Laya Balatro Agent"
        ))
        while True:
            try:
                raw_state = self.client.get_gamestate()
                if raw_state and raw_state.get("state") == "GAME_OVER":
                    console.print(f"\n[bold red]💀 Run Ended (GAME OVER) at Ante {raw_state.get('ante_num', '?')}. Stopping runner to stay at death screen.[/bold red]")
                    break
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
