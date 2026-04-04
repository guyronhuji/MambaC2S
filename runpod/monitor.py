#!/usr/bin/env python3
"""Live training monitor — reads per-job status/CSV files and renders
a Rich table that refreshes in-place every few seconds.

Usage:
    python runpod/monitor.py --log-dir outputs/runpod_run_XXX --jobs 6
"""

import argparse
import csv
import glob
import os
import re
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--jobs",    type=int, default=6, help="Total number of jobs")
    p.add_argument("--refresh", type=float, default=3.0, help="Seconds between updates")
    return p.parse_args()


STATUS_STYLE = {
    "running": "[cyan]▶ run[/cyan]",
    "done":    "[green]✓ done[/green]",
    "failed":  "[red]✗ fail[/red]",
    "queued":  "[dim]· wait[/dim]",
}


def read_job(logfile: str) -> dict:
    name = os.path.basename(logfile).replace(".log", "").replace("_", "/", 1)
    statusfile = logfile.replace(".log", ".status")
    status = open(statusfile).read().strip() if os.path.exists(statusfile) else "queued"

    exp_dir, max_epochs = "", "?"
    try:
        for line in open(logfile):
            if "Experiment ID:" in line and "→" in line:
                exp_dir = line.split("→")[-1].strip()
            m = re.search(r"max_epochs=(\d+)", line)
            if m:
                max_epochs = m.group(1)
    except Exception:
        pass

    epoch = train_loss = val_loss = val_ppl = "-"
    csv_path = f"{exp_dir}/training_log.csv" if exp_dir else ""
    if csv_path and os.path.exists(csv_path):
        try:
            rows = list(csv.DictReader(open(csv_path)))
            if rows:
                last = rows[-1]
                epoch     = last.get("epoch", "-")
                train_loss = f"{float(last['train_loss']):.4f}"  if last.get("train_loss")     else "-"
                val_loss   = f"{float(last['val_loss']):.4f}"    if last.get("val_loss")        else "-"
                val_ppl    = f"{float(last['val_perplexity']):.1f}" if last.get("val_perplexity") else "-"
        except Exception:
            pass

    return dict(name=name, status=status, epoch=epoch,
                max_epochs=max_epochs, train_loss=train_loss,
                val_loss=val_loss, val_ppl=val_ppl)


def build_table(log_dir: str) -> tuple[Table, int, int]:
    logfiles = sorted(glob.glob(f"{log_dir}/*.log"))

    t = Table(box=box.SIMPLE_HEAD, show_footer=False, highlight=True)
    t.add_column("Experiment",  style="bold", min_width=30)
    t.add_column("Status",      min_width=8)
    t.add_column("Epoch",       justify="right", min_width=7)
    t.add_column("Train Loss",  justify="right", min_width=10)
    t.add_column("Val Loss",    justify="right", min_width=10)
    t.add_column("Perplexity",  justify="right", min_width=10)

    done = failed = 0
    for lf in logfiles:
        j = read_job(lf)
        ep = f"{j['epoch']}/{j['max_epochs']}" if j["epoch"] != "-" else "-"
        t.add_row(
            j["name"],
            STATUS_STYLE.get(j["status"], j["status"]),
            ep,
            j["train_loss"],
            j["val_loss"],
            j["val_ppl"],
        )
        if j["status"] == "done":   done += 1
        if j["status"] == "failed": failed += 1

    return t, done, failed


def main():
    args = parse_args()
    console = Console()

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            table, done, failed = build_table(args.log_dir)
            live.update(table)
            if done + failed >= args.jobs:
                break
            time.sleep(args.refresh)

    # Final render outside Live
    table, done, failed = build_table(args.log_dir)
    console.print(table)
    console.print(f"[bold]Complete:[/bold] {done} done, {failed} failed")


if __name__ == "__main__":
    main()
