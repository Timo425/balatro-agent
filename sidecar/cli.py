#!/usr/bin/env python3
"""
Balatro CLI for Agentic AI:
Provides direct, turn-by-turn CLI control for an AI agent to play Balatro itself.
Every command executes an action and automatically prints the resulting gamestate.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sidecar.client import BalatroBotClient, extract_cards_list

console = Console()
SESSION_FILE = ROOT_DIR / ".bench_session.json"

def get_client(endpoint: str = "http://127.0.0.1:12346") -> BalatroBotClient:
    return BalatroBotClient(endpoint=endpoint)

def format_card(c: Dict[str, Any]) -> str:
    r = c.get("value") or c.get("rank") or "?"
    s = c.get("suit") or "?"
    enh = c.get("enhancement") or ""
    enh_str = f" [{enh}]" if enh else ""
    return f"{r} of {s}{enh_str}"

def print_status(client: BalatroBotClient):
    raw_g = client.get_gamestate()
    if not raw_g:
        console.print("[bold red]Viga: BalatroBot server ei vasta pordil 12346.[/bold red]")
        return

    state = raw_g.get("state", "UNKNOWN")
    ante = raw_g.get("ante_num", 1)
    round_num = raw_g.get("round_num", 1)
    money = raw_g.get("money", 0)
    hands_left = raw_g.get("hands_left", 0)
    discards_left = raw_g.get("discards_left", 0)

    round_info = raw_g.get("round", {})
    chips = round_info.get("chips", 0)
    target_chips = round_info.get("target_chips", 0)
    blind_info = raw_g.get("blind", {})
    blind_name = blind_info.get("label") or blind_info.get("name") or "Blind"
    blind_desc = blind_info.get("desc") or ""

    header = (
        f"[bold white]State:[/bold white] [bold cyan]{state}[/bold cyan] | "
        f"[bold white]Ante:[/bold white] [bold yellow]{ante}[/bold yellow] | "
        f"[bold white]Round:[/bold white] [bold yellow]{round_num}[/bold yellow] | "
        f"[bold white]Cash:[/bold white] [bold green]${money}[/bold green] | "
        f"[bold white]Hands:[/bold white] [bold cyan]{hands_left}[/bold cyan] | "
        f"[bold white]Discards:[/bold white] [bold magenta]{discards_left}[/bold magenta] | "
        f"[bold white]Score:[/bold white] [bold yellow]{chips}/{target_chips}[/bold yellow]"
    )
    if blind_desc:
        header += f"\n[dim]{blind_name}: {blind_desc}[/dim]"

    console.print(Panel(header, title="🃏 Balatro Game State", border_style="cyan"))

    # Jokers
    jokers = extract_cards_list(raw_g, "jokers")
    if jokers:
        t_j = Table(title=f"Jokers ({len(jokers)}/5)", show_header=True)
        t_j.add_column("Idx", justify="center", style="cyan")
        t_j.add_column("Name", style="bold yellow")
        t_j.add_column("Effect", style="white")
        t_j.add_column("Sell", justify="right", style="green")
        for i, j in enumerate(jokers):
            val_obj = j.get("value", {}) if isinstance(j, dict) else {}
            eff = val_obj.get("effect") or j.get("effect") or ""
            cost_obj = j.get("cost", {}) if isinstance(j, dict) else {}
            sell_val = cost_obj.get("sell", 0) if isinstance(cost_obj, dict) else 0
            t_j.add_row(str(i), j.get("label") or j.get("key", "?"), eff, f"${sell_val}")
        console.print(t_j)

    # Consumables
    consumables = extract_cards_list(raw_g, "consumables")
    if consumables:
        t_c = Table(title="Consumables", show_header=True)
        t_c.add_column("Idx", justify="center", style="cyan")
        t_c.add_column("Type", style="magenta")
        t_c.add_column("Name", style="bold")
        t_c.add_column("Effect", style="white")
        for i, c in enumerate(consumables):
            val_obj = c.get("value", {}) if isinstance(c, dict) else {}
            eff = val_obj.get("effect") or ""
            t_c.add_row(str(i), c.get("set", "TAROT"), c.get("label") or c.get("key", "?"), eff)
        console.print(t_c)

    # Hand Cards
    if state == "SELECTING_HAND":
        hand_cards = extract_cards_list(raw_g, "hand")
        if hand_cards:
            t_h = Table(title=f"Playing Hand ({len(hand_cards)} cards)", show_header=True)
            t_h.add_column("Idx", justify="center", style="bold cyan")
            t_h.add_column("Card", style="bold yellow")
            t_h.add_column("Suit", style="green")
            t_h.add_column("Rank", style="magenta")
            for i, c in enumerate(hand_cards):
                r = c.get("value") or c.get("rank") or "?"
                s = c.get("suit") or "?"
                t_h.add_row(str(i), format_card(c), s, r)
            console.print(t_h)
            console.print("[dim]Use: `python sidecar/cli.py play 0 1 2 3 4` or `discard 0 1`[/dim]")

    # Shop
    elif state == "SHOP":
        shop_cards = extract_cards_list(raw_g, "shop")
        vouchers = extract_cards_list(raw_g, "vouchers")
        packs = extract_cards_list(raw_g, "packs")
        t_s = Table(title="Shop Items", show_header=True)
        t_s.add_column("Idx", justify="center", style="bold cyan")
        t_s.add_column("Category", style="magenta")
        t_s.add_column("Item", style="bold yellow")
        t_s.add_column("Cost", justify="right", style="green")

        idx = 0
        for c in shop_cards:
            cost = c.get("cost", {}).get("buy", 4) if isinstance(c.get("cost"), dict) else 4
            t_s.add_row(str(idx), c.get("set", "CARD"), c.get("label") or c.get("key", "?"), f"${cost}")
            idx += 1
        for v in vouchers:
            cost = v.get("cost", {}).get("buy", 10) if isinstance(v.get("cost"), dict) else 10
            t_s.add_row(str(idx), "VOUCHER", v.get("label") or v.get("key", "?"), f"${cost}")
            idx += 1
        for p in packs:
            cost = p.get("cost", {}).get("buy", 6) if isinstance(p.get("cost"), dict) else 6
            t_s.add_row(str(idx), "BOOSTER", p.get("label") or p.get("key", "?"), f"${cost}")
            idx += 1
        console.print(t_s)
        console.print("[dim]Use: `python sidecar/cli.py buy <idx>` or `python sidecar/cli.py next` to leave shop[/dim]")

    # Pack opened
    elif state in ("SMODS_BOOSTER_OPENED", "PACK_OPENED"):
        pack_cards = extract_cards_list(raw_g, "pack")
        t_p = Table(title="Booster Pack Contents", show_header=True)
        t_p.add_column("Idx", justify="center", style="bold cyan")
        t_p.add_column("Card", style="bold yellow")
        t_p.add_column("Description", style="white")
        for i, c in enumerate(pack_cards):
            t_p.add_row(str(i), c.get("label") or format_card(c), c.get("value", {}).get("effect", ""))
        console.print(t_p)
        console.print("[dim]Use: `python sidecar/cli.py pick-pack <idx>` or `skip-pack`[/dim]")

    elif state == "ROUND_EVAL":
        console.print("[bold green]Round won! Use: `python sidecar/cli.py cashout`[/bold green]")
    elif state == "BLIND_SELECT":
        console.print("[bold yellow]Blind selection! Use: `python sidecar/cli.py select` or `python sidecar/cli.py skip`[/bold yellow]")
    elif state == "GAME_OVER":
        console.print("[bold red]☠️ GAME OVER! Run has ended.[/bold red]")
        if SESSION_FILE.exists():
            console.print("[cyan]Run `python sidecar/cli.py bench-finish` to save your official BalatroBench score![/cyan]")

def cmd_bench_start(args, client):
    seed = args.seed or "AAAAAAA"
    console.print(f"[bold green]Starting official BalatroBench run:[/bold green] Seed=[yellow]{seed}[/yellow]")

    # Check health / auto-launch
    if not client.health():
        console.print("[yellow]BalatroBot ei vasta, käivitan Balatro automaatselt...[/yellow]")
        from sidecar.bench import BalatroLauncher
        launcher = BalatroLauncher(
            port=12346,
            fast=True,
            headless=False,
            audio=False,
        )
        launcher.start()
        if not launcher.wait_for_health(timeout=45.0):
            console.print("[bold red]Balatro käivitamine ebaõnnestus.[/bold red]")
            return
        time.sleep(5.0)

    # Go to menu first
    client.send_rpc("menu")
    time.sleep(1.0)

    # Start run with seed
    for attempt in range(5):
        res = client.send_rpc("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
        if res:
            break
        time.sleep(2.0)

    time.sleep(1.0)
    # Save session file
    session_data = {
        "seed": seed,
        "start_time": time.time(),
        "deck": "RED",
        "stake": "WHITE",
    }
    SESSION_FILE.write_text(json.dumps(session_data, indent=2))
    console.print(f"[bold green]Run started successfully with seed {seed}![/bold green]")
    print_status(client)

def cmd_bench_finish(args, client):
    raw_g = client.get_gamestate() or {}
    ante = raw_g.get("ante_num", 1)
    round_num = raw_g.get("round_num", 1)
    won = bool(raw_g.get("won", False))
    chips = raw_g.get("round", {}).get("chips", 0)

    start_time = time.time()
    seed = "AAAAAAA"
    if SESSION_FILE.exists():
        try:
            s_data = json.loads(SESSION_FILE.read_text())
            seed = s_data.get("seed", "AAAAAAA")
            start_time = s_data.get("start_time", start_time)
            SESSION_FILE.unlink()
        except Exception:
            pass

    duration = round(time.time() - start_time, 1)

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": "agentic_manual",
        "model": "Antigravity/Claude",
        "deck": "RED",
        "stake": "WHITE",
        "results": [{
            "seed": seed,
            "ante_reached": ante,
            "rounds_reached": round_num,
            "final_score": chips,
            "won": won,
            "duration_seconds": duration,
        }]
    }

    out_file = ROOT_DIR / "benchmark_results.json"
    out_file.write_text(json.dumps(result, indent=2))
    console.print(Panel(
        f"[bold green]BalatroBench Official Results Saved![/bold green]\n"
        f"• Seed: [yellow]{seed}[/yellow]\n"
        f"• Ante Reached: [bold yellow]{ante}[/bold yellow] / 8\n"
        f"• Rounds Reached: [bold magenta]{round_num}[/bold magenta] / 24\n"
        f"• Final Chips: [bold cyan]{chips}[/bold cyan]\n"
        f"• Won: [bold {'green' if won else 'red'}]{won}[/bold]\n"
        f"• Duration: {duration}s\n"
        f"Saved to: [dim]{out_file}[/dim]",
        title="🏆 BalatroBench Scorecard",
        border_style="green"
    ))

def main():
    parser = argparse.ArgumentParser(description="Balatro Agentic CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("status", help="Print current gamestate")
    subparsers.add_parser("cashout", help="Cash out round reward")
    subparsers.add_parser("next", help="Leave shop (next round)")
    subparsers.add_parser("skip", help="Skip current blind")
    subparsers.add_parser("select", help="Select / play current blind")

    p_play = subparsers.add_parser("play", help="Play cards by hand indices")
    p_play.add_argument("indices", type=int, nargs="+", help="Card indices (0-indexed)")

    p_discard = subparsers.add_parser("discard", help="Discard cards by hand indices")
    p_discard.add_argument("indices", type=int, nargs="+", help="Card indices (0-indexed)")

    p_buy = subparsers.add_parser("buy", help="Buy item from shop by index")
    p_buy.add_argument("index", type=int, help="Shop item index")

    p_sell = subparsers.add_parser("sell", help="Sell joker by index")
    p_sell.add_argument("index", type=int, help="Joker index")

    p_use = subparsers.add_parser("use", help="Use consumable card")
    p_use.add_argument("index", type=int, help="Consumable index")
    p_use.add_argument("targets", type=int, nargs="*", help="Optional hand card target indices")

    p_rearrange = subparsers.add_parser("rearrange", help="Rearrange jokers order")
    p_rearrange.add_argument("order", type=int, nargs="+", help="New permutation of indices")

    p_pack = subparsers.add_parser("pick-pack", help="Pick card from booster pack")
    p_pack.add_argument("index", type=int, help="Pack card index")
    subparsers.add_parser("skip-pack", help="Skip booster pack")

    p_bench = subparsers.add_parser("bench-start", help="Start official BalatroBench run")
    p_bench.add_argument("--seed", default="AAAAAAA", help="Seed (default: AAAAAAA)")

    subparsers.add_parser("bench-finish", help="Finish and save official BalatroBench score")

    args = parser.parse_args()
    client = get_client()

    if not args.command or args.command == "status":
        print_status(client)
        return

    if args.command == "bench-start":
        cmd_bench_start(args, client)
        return
    elif args.command == "bench-finish":
        cmd_bench_finish(args, client)
        return
    elif args.command == "cashout":
        client.cash_out()
    elif args.command == "next":
        client.next_round()
    elif args.command == "select":
        client.select_blind()
    elif args.command == "skip":
        client.skip_blind()
    elif args.command == "play":
        client.play_cards(args.indices)
    elif args.command == "discard":
        client.discard_cards(args.indices)
    elif args.command == "buy":
        raw_g = client.get_gamestate() or {}
        shop_cards = extract_cards_list(raw_g, "shop")
        vouchers = extract_cards_list(raw_g, "vouchers")
        idx = args.index
        if idx < len(shop_cards):
            client.buy_card(idx)
        elif idx < len(shop_cards) + len(vouchers):
            client.buy_voucher(idx - len(shop_cards))
        else:
            pack_idx = idx - len(shop_cards) - len(vouchers)
            client.buy_pack(pack_idx)
    elif args.command == "sell":
        client.sell_joker(args.index)
    elif args.command == "use":
        targets = args.targets if args.targets else None
        client.use_consumable(args.index, target_cards=targets)
    elif args.command == "rearrange":
        client.rearrange_jokers(args.order)
    elif args.command == "pick-pack":
        client.select_pack_card(card_idx=args.index)
    elif args.command == "skip-pack":
        client.select_pack_card(skip=True)

    time.sleep(0.5)
    print_status(client)

if __name__ == "__main__":
    main()
