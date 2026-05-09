#!/usr/bin/env python3
"""
Terminal dashboard for the Music Intelligence Agent.

Usage:
  uv run python dashboard.py           # single snapshot
  uv run python dashboard.py --watch   # live refresh every 5s
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text
from rich import box

DB_PATH = Path(__file__).parent / "agent.db"
LOG_PATH = Path(__file__).parent / "agent.log"
REFRESH_SECONDS = 5


def _read_db():
    import sqlite3
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    status_rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM songs GROUP BY status"
    ).fetchall()
    status = {r["status"]: r["cnt"] for r in status_rows}

    recent = conn.execute(
        """SELECT s.artist, s.title, s.analyzed_at, sf.bpm, sf.key, sf.energy
           FROM songs s
           LEFT JOIN song_features sf ON s.id = sf.song_id
           WHERE s.status = 'done'
           ORDER BY s.analyzed_at DESC LIMIT 10"""
    ).fetchall()

    discovery = conn.execute(
        """SELECT source_artist, COUNT(*) as cnt
           FROM discovery_log GROUP BY source_artist ORDER BY cnt DESC LIMIT 5"""
    ).fetchall()

    key_rows = conn.execute(
        "SELECT key, COUNT(*) as cnt FROM song_features WHERE key IS NOT NULL GROUP BY key ORDER BY cnt DESC LIMIT 8"
    ).fetchall()

    bpm_row = conn.execute(
        "SELECT MIN(bpm) as bpm_min, MAX(bpm) as bpm_max, AVG(bpm) as bpm_avg FROM song_features WHERE bpm IS NOT NULL"
    ).fetchone()

    genre_rows = conn.execute(
        "SELECT genres FROM song_features WHERE genres IS NOT NULL"
    ).fetchall()

    conn.close()

    genre_counter: Counter = Counter()
    for (g,) in genre_rows:
        for genre in json.loads(g or "[]"):
            genre_counter[genre] += 1

    return {
        "status": status,
        "recent": [dict(r) for r in recent],
        "discovery": [dict(r) for r in discovery],
        "top_keys": [dict(r) for r in key_rows],
        "bpm": (bpm_row["bpm_min"], bpm_row["bpm_max"],
                round(bpm_row["bpm_avg"], 1) if bpm_row["bpm_avg"] else None),
        "top_genres": genre_counter.most_common(6),
    }


def _read_log_tail(n: int = 6) -> list[str]:
    if not LOG_PATH.exists():
        return ["(agent.log not found)"]
    with open(LOG_PATH) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def _build_display(data: dict) -> list:
    renderables = []

    # ── Library Stats ──
    s = data["status"]
    total = sum(s.values())
    done = s.get("done", 0)
    pct = f"{100*done//total}%" if total else "0%"
    stats_text = (
        f"[green]Done: {done}[/green]  "
        f"[yellow]Pending: {s.get('pending', 0)}[/yellow]  "
        f"[blue]Analyzing: {s.get('analyzing', 0)}[/blue]  "
        f"[red]Error: {s.get('error', 0)}[/red]  "
        f"[dim]{pct} complete[/dim]"
    )
    renderables.append(Panel(stats_text, title="[bold]Library Stats[/bold]", box=box.ROUNDED))

    # ── Agent Log ──
    log_lines = _read_log_tail()
    log_text = "\n".join(log_lines) or "(no log entries)"
    renderables.append(Panel(log_text, title="[bold]Agent Log (last 6 lines)[/bold]", box=box.ROUNDED))

    # ── Recent Analyses ──
    if data["recent"]:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Artist", max_width=20)
        table.add_column("Title", max_width=25)
        table.add_column("BPM", justify="right", max_width=6)
        table.add_column("Key", max_width=12)
        table.add_column("Energy", max_width=8)
        table.add_column("Analyzed At", max_width=20)
        for r in data["recent"]:
            table.add_row(
                str(r.get("artist") or "?")[:20],
                str(r.get("title") or "?")[:25],
                str(r.get("bpm") or ""),
                str(r.get("key") or ""),
                str(r.get("energy") or ""),
                str(r.get("analyzed_at") or "")[:19],
            )
        renderables.append(Panel(table, title="[bold]Recent Analyses[/bold]", box=box.ROUNDED))

    # ── Keys + BPM + Genres side by side ──
    side_panels = []

    if data["top_keys"]:
        key_table = Table(box=box.SIMPLE, show_header=False)
        key_table.add_column("Key")
        key_table.add_column("Bar")
        key_table.add_column("N", justify="right")
        max_cnt = max(r["cnt"] for r in data["top_keys"]) or 1
        for r in data["top_keys"]:
            bar = "█" * int(10 * r["cnt"] / max_cnt)
            key_table.add_row(r["key"] or "?", f"[cyan]{bar}[/cyan]", str(r["cnt"]))
        side_panels.append(Panel(key_table, title="[bold]Top Keys[/bold]", box=box.ROUNDED))

    bpm_min, bpm_max, bpm_avg = data["bpm"]
    if bpm_avg is not None:
        bpm_text = f"Min: {bpm_min}\nMax: {bpm_max}\nAvg: {bpm_avg}"
        side_panels.append(Panel(bpm_text, title="[bold]BPM Range[/bold]", box=box.ROUNDED))

    if data["top_genres"]:
        genre_table = Table(box=box.SIMPLE, show_header=False)
        genre_table.add_column("Genre")
        genre_table.add_column("N", justify="right")
        for genre, cnt in data["top_genres"]:
            genre_table.add_row(genre, str(cnt))
        side_panels.append(Panel(genre_table, title="[bold]Genres[/bold]", box=box.ROUNDED))

    if side_panels:
        renderables.append(Columns(side_panels))

    return renderables


def render_once(console: Console):
    data = _read_db()
    if data is None:
        console.print("[red]agent.db not found — is the agent running?[/red]")
        return
    for r in _build_display(data):
        console.print(r)


def render_watch(console: Console):
    from rich.console import Group
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            data = _read_db()
            if data is None:
                live.update(Panel("[red]agent.db not found[/red]"))
            else:
                renderables = _build_display(data)
                live.update(Group(*renderables))
            time.sleep(REFRESH_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="Music Intelligence Agent Dashboard")
    parser.add_argument("--watch", action="store_true", help="Live refresh every 5s")
    args = parser.parse_args()

    console = Console()
    if args.watch:
        render_watch(console)
    else:
        render_once(console)


if __name__ == "__main__":
    main()
