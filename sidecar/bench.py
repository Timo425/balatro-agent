"""
BalatroBench Benchmark Runner:
Automates evaluation of AI agents across official BalatroBench fixed seeds.

Official BalatroBench Specifications:
- Deck: Red Deck (RED)
- Stake: White Stake (WHITE)
- Seeds: AAAAAAA, BBBBBBB, CCCCCCC, DDDDDDD, EEEEEEE
- Success Metric: Average Rounds Reached (out of 24 rounds for Ante 8 Boss Blind defeat)

Automated Game Launch:
- Detects Steam/Proton/Balatro paths automatically
- Launches Balatro.exe via Proton with BalatroBot mod env vars
- Waits for health endpoint before starting runs
- Cleans up via wineserver -k after completion
"""

import argparse
import json
import os
import subprocess
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

BALATRO_APP_ID = "2379780"

# ---------------------------------------------------------------------------
# Steam / Proton automaatne tuvastamine
# ---------------------------------------------------------------------------

def detect_steam_root() -> Optional[Path]:
    """Otsib Steam installatsioonitekauste."""
    home = Path.home()
    candidates = [
        home / ".local/share/Steam",
        home / ".steam/steam",
        home / ".steam/debian-installation",
    ]
    for p in candidates:
        if (p / "steamapps").is_dir():
            return p
    return None


def detect_proton_path(steam_root: Path) -> Optional[Path]:
    """Leiab esimese saadaoleva Proton executable."""
    common = steam_root / "steamapps/common"
    if not common.is_dir():
        return None
    for d in sorted(common.iterdir(), reverse=True):
        proton = d / "proton"
        if proton.is_file() and "proton" in d.name.lower():
            return proton
    return None


def detect_compat_data(steam_root: Path) -> Optional[Path]:
    """Leiab Balatro compat data kausta."""
    p = steam_root / f"steamapps/compatdata/{BALATRO_APP_ID}"
    return p if p.is_dir() else None


def detect_balatro_exe(steam_root: Path) -> Optional[Path]:
    """Leiab Balatro.exe."""
    exe = steam_root / "steamapps/common/Balatro/Balatro.exe"
    return exe if exe.is_file() else None


class BalatroLauncher:
    """Kaevitab Balatro otse Protoni kaudu (ilma Steam kliendita).

    Kasutab sama meetodit mis ametlik BalatroBot Python launcher:
    proton run Balatro.exe koos vajalike env varidega.
    """

    def __init__(
        self,
        port: int = 12346,
        headless: bool = False,
        fast: bool = True,
        audio: bool = False,
        gamespeed: int = 10,
    ):
        self.port = port
        self.headless = headless
        self.fast = fast
        self.audio = audio
        self.gamespeed = gamespeed
        self.process: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None

        # Automaatne tuvastamine
        self.steam_root = detect_steam_root()
        if not self.steam_root:
            raise RuntimeError(
                "Steam installatsiooni ei leitud. "
                "Otsiti: ~/.local/share/Steam, ~/.steam/steam, ~/.steam/debian-installation"
            )

        self.proton_path = detect_proton_path(self.steam_root)
        if not self.proton_path:
            raise RuntimeError("Proton executable ei leitud Steam common kaustas.")

        self.compat_data = detect_compat_data(self.steam_root)
        if not self.compat_data:
            raise RuntimeError(f"Balatro compat data ei leitud (app {BALATRO_APP_ID}).")

        self.balatro_exe = detect_balatro_exe(self.steam_root)
        if not self.balatro_exe:
            raise RuntimeError("Balatro.exe ei leitud Steam common/Balatro kaustas.")

        # Lovely (version.dll) kontroll
        lovely_dll = self.balatro_exe.parent / "version.dll"
        if not lovely_dll.is_file():
            console.print("[yellow]version.dll (Lovely injector) ei leitud - modid ei pruugi laadida![/yellow]")

    def _build_env(self) -> Dict[str, str]:
        """Ehitab env dict Protoni ja BalatroBot jaoks."""
        env = os.environ.copy()

        # Proton nouded
        env["WINEDLLOVERRIDES"] = "version=n,b"
        env["PROTON_ENABLE_NVAPI"] = "1"
        env["STEAM_COMPAT_CLIENT_INSTALL_PATH"] = str(self.steam_root)
        env["STEAM_COMPAT_DATA_PATH"] = str(self.compat_data)

        # BalatroBot mod seadistused
        env["BALATROBOT_PORT"] = str(self.port)
        env["BALATROBOT_HOST"] = "127.0.0.1"

        if self.fast:
            env["BALATROBOT_FAST"] = "1"
        if self.headless:
            env["BALATROBOT_HEADLESS"] = "1"
        if not self.audio:
            env["BALATROBOT_AUDIO"] = "0"

        env["BALATROBOT_GAMESPEED"] = str(self.gamespeed)

        return env

    def start(self) -> None:
        """Kaevitab Balatro Protoni kaudu."""
        if self.process is not None:
            raise RuntimeError("Balatro on juba kaevitatud!")

        env = self._build_env()
        cmd = [str(self.proton_path), "run", str(self.balatro_exe)]

        # Logi fail
        log_dir = ROOT_DIR / "logs" / "bench"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"balatro_{timestamp}_port{self.port}.log"

        console.print(Panel.fit(
            f"[bold cyan]Kaevitan Balatro...[/bold cyan]\n"
            f"Proton: [dim]{self.proton_path}[/dim]\n"
            f"Exe: [dim]{self.balatro_exe}[/dim]\n"
            f"Port: [yellow]{self.port}[/yellow] | "
            f"Fast: {'[green]YES[/green]' if self.fast else '[red]NO[/red]'} | "
            f"Headless: {'[green]YES[/green]' if self.headless else '[red]NO[/red]'} | "
            f"Audio: {'[green]YES[/green]' if self.audio else '[red]NO[/red]'}",
            border_style="cyan"
        ))

        with open(self.log_path, "w") as log_file:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        console.print(f"[dim]PID: {self.process.pid} | Logi: {self.log_path}[/dim]")

    def wait_for_health(self, timeout: float = 60.0) -> bool:
        """Ootab kuni BalatroBot health endpoint vastab."""
        import httpx

        url = f"http://127.0.0.1:{self.port}"
        payload = {"jsonrpc": "2.0", "method": "health", "params": {}, "id": 1}
        start_time = time.time()

        console.print(f"[bold yellow]Ootan BalatroBot serverit pordil {self.port}...[/bold yellow]")

        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                console.print(f"[bold red]Balatro protsess suri (exit code: {self.process.returncode})[/bold red]")
                if self.log_path:
                    console.print(f"[dim]Vaata logi: {self.log_path}[/dim]")
                return False

            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.post(url, json=payload)
                    data = resp.json()
                    if "result" in data and data["result"].get("status") == "ok":
                        elapsed = round(time.time() - start_time, 1)
                        console.print(f"[bold green]BalatroBot server vastab! ({elapsed}s)[/bold green]")
                        return True
            except (httpx.ConnectError, httpx.TimeoutException, Exception):
                pass

            time.sleep(0.5)

        console.print(f"[bold red]Health check timeout ({timeout}s)[/bold red]")
        return False

    def stop(self) -> None:
        """Peatab Balatro ja puhastab Wine protsessid."""
        if self.process is None:
            return

        console.print("[bold yellow]Peatan Balatro...[/bold yellow]")

        # wineserver -k puhastab koik Wine protsessid selles prefixis
        wineserver = self.proton_path.parent / "files" / "bin" / "wineserver"
        if wineserver.is_file() and self.compat_data:
            wineprefix = self.compat_data / "pfx"
            if wineprefix.is_dir():
                try:
                    subprocess.run(
                        [str(wineserver), "-k"],
                        env={"WINEPREFIX": str(wineprefix)},
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            console.print("[yellow]Force kill...[/yellow]")
            self.process.kill()
            self.process.wait()
        except Exception:
            pass

        self.process = None
        console.print("[bold green]Balatro peatatud.[/bold green]")


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

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
        auto_launch: bool = False,
        headless: bool = False,
        fast: bool = True,
        stay_on_death: bool = True,
        node: str = "choice",
    ):
        self.endpoint = endpoint
        self.backend_type = backend
        self.model = model
        self.deck = deck
        self.stake = stake
        self.auto_launch = auto_launch
        self.stay_on_death = stay_on_death
        self.launcher: Optional[BalatroLauncher] = None

        if auto_launch:
            port = int(endpoint.split(":")[-1])
            self.launcher = BalatroLauncher(
                port=port,
                headless=headless,
                fast=fast,
                audio=False,
            )

        backend_obj = create_backend(
            backend_type=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self.node = node
        self.agent = BalatroAgent(
            backend=backend_obj,
            poll_mode=(node == "poll" and getattr(backend_obj, "is_tree", False)),
        )
        self.client = BalatroBotClient(endpoint=endpoint, agent=self.agent)

    def ensure_game_running(self) -> bool:
        """Ensures Balatro is running and BalatroBot is responding."""
        if self.client.health():
            console.print("[bold green]BalatroBot is already running![/bold green]")
            return True

        if not self.launcher:
            console.print(
                "[bold red]BalatroBot is not responding and auto-launch is disabled.[/bold red]\n"
                "[yellow]Launch Balatro manually or use the --auto-launch flag.[/yellow]"
            )
            return False

        self.launcher.start()
        if not self.launcher.wait_for_health(timeout=60.0):
            return False

        # Wait for menu UI to fully load
        console.print("[yellow]Waiting for menu UI to load completely...[/yellow]")
        time.sleep(8.0)
        return True

    def ensure_at_menu(self) -> bool:
        """Ensures game is in MENU state (required by start endpoint)."""
        state = self.client.get_gamestate()
        if not state:
            return False

        current = state.get("state", "")
        if current == "MENU":
            return True

        console.print(f"[yellow]Current state: {current} -> Navigating to menu...[/yellow]")
        result = self.client.send_rpc("menu")
        if result is None:
            console.print("[red]Menu RPC failed[/red]")
            return False

        for _ in range(20):
            time.sleep(0.5)
            state = self.client.get_gamestate()
            if state and state.get("state") == "MENU":
                return True

        console.print("[red]Failed to navigate to menu[/red]")
        return False

    def start_seed_run(self, seed: str) -> bool:
        """Starts a fresh game run with the specified seed."""
        console.print(
            f"[bold cyan]BalatroBench run:[/bold cyan] "
            f"Seed=[yellow]{seed}[/yellow], Deck={self.deck}, Stake={self.stake}"
        )

        if not self.ensure_at_menu():
            console.print("[bold red]Cannot navigate to menu[/bold red]")
            return False

        start_params = {
            "deck": self.deck,
            "stake": self.stake,
            "seed": seed,
        }

        # Retry logic: menu UI might take a moment to be ready
        for attempt in range(5):
            res = self.client.send_rpc("start", start_params)
            if res:
                time.sleep(0.5)
                return True
            console.print(f"[yellow]Start attempt {attempt + 1}/5 failed, waiting 3s...[/yellow]")
            time.sleep(3.0)

        console.print(f"[bold red]Start RPC failed all 5 attempts (seed {seed})[/bold red]")
        return False

    def run_single_benchmark(self, seed: str) -> Dict[str, Any]:
        """Plays a full run until GAME_OVER or victory."""
        started = self.start_seed_run(seed)
        if not started:
            return {
                "seed": seed,
                "error": "Failed to initialize run",
                "rounds_reached": 0,
                "ante_reached": 0,
                "final_score": 0,
                "won": False,
                "money": 0,
                "duration_seconds": 0,
                "backend": self.backend_type,
                "model": self.model or "default",
            }

        runner = BalatroLayaRunner(endpoint=self.endpoint, agent=self.agent)

        # The agent is shared across seeds, so record the counters at the start
        # and report this run's own deltas rather than a running total.
        before = self.agent.backend.telemetry()
        start_time = time.time()
        last_ante = 1
        last_round = 1
        last_score = 0
        is_won = False
        final_money = 0

        while True:
            raw_g = runner.client.get_gamestate()
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

            try:
                runner.run_step()
            except Exception as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")

            time.sleep(0.1)

        elapsed = time.time() - start_time
        after = self.agent.backend.telemetry()
        telemetry = {
            "calls": after.get("calls", 0) - before.get("calls", 0),
            "fallbacks": after.get("fallbacks", 0) - before.get("fallbacks", 0),
        }
        if telemetry["fallbacks"]:
            console.print(
                f"[bold red]⚠ {telemetry['fallbacks']} fallback decision(s)[/bold red] -- "
                "the model did not answer and Python picked. This result does not "
                "measure the model alone."
            )
        return {
            "seed": seed,
            "backend": self.backend_type,
            "node": self.node,
            "decisions": telemetry.get("calls", 0),
            "fallbacks": telemetry.get("fallbacks", 0),
            "model": self.model or "default",
            "ante_reached": last_ante,
            "rounds_reached": last_round,
            "final_score": last_score,
            "won": is_won,
            "money": final_money,
            "duration_seconds": round(elapsed, 1),
        }

    def run_suite(
        self,
        seeds: Optional[List[str]] = None,
        runs_per_seed: int = 1,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Runs the complete benchmark suite across seeds."""
        target_seeds = seeds or OFFICIAL_BENCHMARK_SEEDS
        results = []

        if not self.ensure_game_running():
            console.print("[bold red]Game failed to launch, aborting.[/bold red]")
            return results

        console.print(Panel.fit(
            f"[bold green]BalatroBench Automated Evaluation Suite[/bold green]\n"
            f"Backend: [cyan]{self.backend_type}[/cyan] | Model: [cyan]{self.model or 'default'}[/cyan]\n"
            f"Seeds: {len(target_seeds)} ({','.join(target_seeds)}) | Runs per seed: {runs_per_seed}",
            border_style="green"
        ))

        try:
            for seed in target_seeds:
                for run_idx in range(runs_per_seed):
                    console.print(f"\n[bold blue]--- Seed {seed} (Run {run_idx + 1}/{runs_per_seed}) ---[/bold blue]")
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
                    if not res.get("won") and self.stay_on_death:
                        console.print("\n[bold red]💀 Run Ended (GAME OVER). Stopping so you can inspect the death screen.[/bold red]")
                        return results
        finally:
            if self.launcher and not self.stay_on_death:
                self.launcher.stop()

        self.print_summary(results)

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "backend": self.backend_type,
                    "model": self.model,
                    "deck": self.deck,
                    "stake": self.stake,
                    "results": results,
                }, f, indent=2)
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

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
                f"Average Rounds Reached: [bold magenta]{avg_rounds} / 24[/bold magenta]\n"
                f"Average Ante Reached: [bold yellow]{avg_ante} / 8[/bold yellow]\n"
                f"Win Rate: [bold green]{win_rate}%[/bold green]\n"
                f"Total Runs: {n}",
                title="Leaderboard Score",
                border_style="magenta"
            ))


def main():
    parser = argparse.ArgumentParser(
        description="BalatroBench Official Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Balatro automatically + run one seed (hierarchical Laya tree)
  PYTHONPATH=. .venv/bin/python sidecar/bench.py --auto-launch --seeds AAAAAAA

  # All 5 official seeds, headless mode
  PYTHONPATH=. .venv/bin/python sidecar/bench.py --auto-launch --headless

  # API LLM backend (GPT-4o), game already running
  export OPENAI_API_KEY="sk-..."
  PYTHONPATH=. .venv/bin/python sidecar/bench.py --backend api_llm --model gpt-4o

  # Save results to file
  PYTHONPATH=. .venv/bin/python sidecar/bench.py --auto-launch --output results.json
"""
    )
    parser.add_argument("--backend", default="laya_tree",
                        choices=["laya", "laya_tree", "api_llm", "api_llm_tree", "agentic"],
                        help="Decision backend (default: laya_tree, the hierarchical Laya tree)")
    parser.add_argument("--node", default="choice", choices=["choice", "poll"],
                        help="Tree node shape: one choice per node (default), or a "
                             "batched yes/no poll over the options")
    parser.add_argument("--model", default=None,
                        help="Model identifier (e.g. gpt-4o, claude-3-5-sonnet)")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default: AAAAAAA..EEEEEEE)")
    parser.add_argument("--deck", default="RED",
                        help="Deck name (default: RED)")
    parser.add_argument("--stake", default="WHITE",
                        help="Stake name (default: WHITE)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per seed (default: 1)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346",
                        help="BalatroBot JSON-RPC URL")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Path to save output results JSON")

    launch_group = parser.add_argument_group("Auto-Launch")
    launch_group.add_argument("--auto-launch", action="store_true",
                              help="Launch Balatro automatically via Proton")
    launch_group.add_argument("--headless", action="store_true",
                              help="Headless mode (minimal window, no rendering)")
    launch_group.add_argument("--no-fast", action="store_true",
                              help="Disable fast mode (default: fast is ON)")
    launch_group.add_argument("--no-stay-on-death", action="store_true",
                              help="Automatically navigate away from death screen instead of staying on it")

    args = parser.parse_args()

    seed_list = [s.strip() for s in args.seeds.split(",")] if args.seeds else OFFICIAL_BENCHMARK_SEEDS
    fast = not args.no_fast
    stay_on_death = not args.no_stay_on_death

    bench = BalatroBenchmark(
        endpoint=args.endpoint,
        backend=args.backend,
        model=args.model,
        deck=args.deck,
        stake=args.stake,
        auto_launch=args.auto_launch,
        headless=args.headless,
        fast=fast,
        stay_on_death=stay_on_death,
        node=args.node,
    )

    bench.run_suite(seeds=seed_list, runs_per_seed=args.runs, output_file=args.output)


if __name__ == "__main__":
    main()
