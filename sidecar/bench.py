"""
BalatroBench Benchmark Runner:
Automates evaluation of AI agents across official BalatroBench fixed seeds.

Official BalatroBench Specifications:
- Deck: Red Deck (RED)
- Stake: White Stake (WHITE)
- Seeds: AAAAAAA, BBBBBBB, CCCCCCC, DDDDDDD, EEEEEEE
- Success Metric: Average Rounds Reached (out of 24 rounds for Ante 8 Boss Blind defeat)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sidecar.client import BalatroBotClient
from sidecar.agent import BalatroAgent
from sidecar.runner import BalatroLayaRunner
from sidecar.backends import create_backend

console = Console()

OFFICIAL_BENCHMARK_SEEDS = [
    "AAAAAAA",
    "BBBBBBB",
    "CCCCCCC",
    "DDDDDDD",
    "EEEEEEE",
]

class BalatroBenchmark:
    """Automates BalatroBench fixed-seed benchmarking runs."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:12346",
        backend: str = "laya",
        model: Optional[str] = None,
        deck: str = "RED",
        stake: str = "WHITE",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.backend_type = backend
        self.model = model
        self.deck = deck
        self.stake = stake

        # Instantiate agent with requested backend
        backend_obj = create_backend(
            backend_type=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self.agent = BalatroAgent(backend=backend_obj)
        self.client = BalatroBotClient(endpoint=endpoint, agent=self.agent)

    def start_seed_run(self, seed: str) -> bool:
        """Starts a fresh game run with the specified seed."""
        console.print(f"[bold cyan]🚀 Starting BalatroBench run:[/bold cyan] Seed=[yellow]{seed}[/yellow], Deck={self.deck}, Stake={self.stake}")

        # If currently in game, ensure we return to menu or restart
        try:
            state_resp = self.client.get_state()
            curr_state = (state_resp.get("state") or "") if state_resp else ""
            if curr_state != "MENU":
                self.client.menu()
                time.sleep(0.5)
        except Exception:
            pass

        # Send start command with seed
        start_params = {
            "deck": self.deck,
            "stake": self.stake,
            "seed": seed,
        }
        res = self.client.send_rpc("start", start_params)
        if not res:
            console.print(f"[bold red]❌ Failed to start run with seed {seed}[/bold red]")
            return False

        time.sleep(0.5)
        return True

    def run_single_benchmark(self, seed: str, max_rounds: int = 40) -> Dict[str, Any]:
        """Plays a full run until GAME_OVER or victory (won=True)."""
        started = self.start_seed_run(seed)
        if not started:
            return {
                "seed": seed,
                "error": "Failed to initialize run",
                "rounds_reached": 0,
                "ante_reached": 0,
                "won": False,
            }

        runner = BalatroLayaRunner(endpoint=self.endpoint, agent=self.agent)

        start_time = time.time()
        last_ante = 1
        last_round = 1
        last_score = 0
        is_won = False
        final_money = 0

        # Run the game loop
        while True:
            raw_g = runner.client.get_state()
            if not raw_g:
                time.sleep(0.5)
                continue

            state_name = raw_g.get("state")
            if "ante_num" in raw_g and raw_g["ante_num"] is not None:
                last_ante = raw_g["ante_num"]
            if "round_num" in raw_g and raw_g["round_num"] is not None:
                last_round = raw_g["round_num"]
            if "money" in raw_g and raw_g["money"] is not None:
                final_money = raw_g["money"]

            is_won = bool(raw_g.get("won", False))

            if state_name == "GAME_OVER" or is_won:
                round_info = raw_g.get("round", {})
                last_score = round_info.get("chips", 0)
                break

            # Handle current state
            try:
                runner.step()
            except Exception as e:
                console.print(f"[yellow]Warning during step: {e}[/yellow]")

            time.sleep(0.1)

        elapsed = time.time() - start_time
        result = {
            "seed": seed,
            "backend": self.backend_type,
            "model": self.model or "default",
            "ante_reached": last_ante,
            "rounds_reached": last_round,
            "final_score": last_score,
            "won": is_won,
            "money": final_money,
            "duration_seconds": round(elapsed, 1),
        }
        return result

    def run_suite(
        self,
        seeds: Optional[List[str]] = None,
        runs_per_seed: int = 1,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Runs the complete benchmark suite across seeds."""
        target_seeds = seeds or OFFICIAL_BENCHMARK_SEEDS
        results = []

        console.print(Panel.fit(
            f"[bold green]BalatroBench Automated Evaluation Suite[/bold green]\n"
            f"Backend: [cyan]{self.backend_type}[/cyan] | Model: [cyan]{self.model or 'default'}[/cyan]\n"
            f"Seeds: {len(target_seeds)} ({chr(44).join(target_seeds)}) | Runs per seed: {runs_per_seed}",
            border_style="green"
        ))

        for seed in target_seeds:
            for run_idx in range(runs_per_seed):
                console.print(f"\n[bold blue]━━━ Seed {seed} (Run {run_idx + 1}/{runs_per_seed}) ━━━[/bold blue]")
                res = self.run_single_benchmark(seed)
                res["run_index"] = run_idx + 1
                results.append(res)
                console.print(
                    f"Result: Ante [bold]{res['ante_reached']}[/bold] | "
                    f"Round [bold]{res['rounds_reached']}[/bold] | "
                    f"Score: {res['final_score']} | "
                    f"Won: {res['won']} | "
                    f"Time: {res['duration_seconds']}s"
                )

        # Print summary table
        self.print_summary(results)

        # Save to output file if requested
        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "backend": self.backend_type,
                    "model": self.model,
                    "results": results,
                }, f, indent=2)
            console.print(f"[bold green]Saved benchmark results to {output_file}[/bold green]")

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Prints a Rich table summarizing the benchmark run."""
        table = Table(title="BalatroBench Benchmark Results", border_style="cyan")
        table.add_column("Seed", style="bold yellow")
        table.add_column("Run", justify="center")
        table.add_column("Ante", justify="center")
        table.add_column("Round", justify="center", style="bold magenta")
        table.add_column("Chips", justify="right")
        table.add_column("Won?", justify="center")
        table.add_column("Time (s)", justify="right")

        total_rounds = 0
        total_antes = 0
        wins = 0

        for r in results:
            table.add_row(
                str(r.get("seed")),
                str(r.get("run_index", 1)),
                str(r.get("ante_reached", 0)),
                str(r.get("rounds_reached", 0)),
                f"{r.get('final_score', 0):,}",
                "[green]WIN[/green]" if r.get("won") else "[red]LOSS[/red]",
                str(r.get("duration_seconds", 0)),
            )
            total_rounds += r.get("rounds_reached", 0)
            total_antes += r.get("ante_reached", 0)
            if r.get("won"):
                wins += 1

        console.print(table)

        n = len(results)
        if n > 0:
            avg_rounds = round(total_rounds / n, 2)
            avg_ante = round(total_antes / n, 2)
            win_rate = round((wins / n) * 100, 1)

            console.print(Panel(
                f"[bold white]Benchmark Summary Statistics:[/bold white]\n"
                f"• Average Rounds Reached: [bold magenta]{avg_rounds} / 24[/bold magenta]\n"
                f"• Average Ante Reached: [bold yellow]{avg_ante} / 8[/bold yellow]\n"
                f"• Win Rate: [bold green]{win_rate}%[/bold green]\n"
                f"• Total Runs: {n}",
                title="Leaderboard Score",
                border_style="magenta"
            ))


def main():
    parser = argparse.ArgumentParser(description="BalatroBench Official Benchmark Runner")
    parser.add_argument("--backend", default="laya", choices=["laya", "api_llm", "agentic"], help="Decision backend to test")
    parser.add_argument("--model", default=None, help="Model identifier (e.g. gpt-4o, claude-3-5-sonnet, convaiinnovations/laya)")
    parser.add_argument("--seeds", default=None, help="Comma-separated list of seeds (default: official AAAAAAA..EEEEEEE)")
    parser.add_argument("--deck", default="RED", help="Deck name (default: RED)")
    parser.add_argument("--stake", default="WHITE", help="Stake name (default: WHITE)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per seed")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="BalatroBot JSON-RPC URL")
    parser.add_argument("--output", default="benchmark_results.json", help="Path to save output results JSON")
    args = parser.parse_args()

    seed_list = [s.strip() for s in args.seeds.split(",")] if args.seeds else OFFICIAL_BENCHMARK_SEEDS

    bench = BalatroBenchmark(
        endpoint=args.endpoint,
        backend=args.backend,
        model=args.model,
        deck=args.deck,
        stake=args.stake,
    )
    bench.run_suite(seeds=seed_list, runs_per_seed=args.runs, output_file=args.output)


if __name__ == "__main__":
    main()
