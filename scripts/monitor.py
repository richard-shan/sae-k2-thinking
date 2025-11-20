#!/usr/bin/env python3
"""
Monitoring utility for activation collection runs.

Reimplements the original shell-based monitor using Python so it can be
executed in environments where shell scripts are not permitted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


def clear_screen() -> None:
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_int(value: int) -> str:
    return f"{value:,}"


def shard_status(shard_id: int) -> str:
    checkpoint = load_json(REPO_ROOT / "data" / f"shard{shard_id}" / "checkpoint.json")
    metadata = load_json(REPO_ROOT / "data" / f"shard{shard_id}" / "metadata.json")

    if checkpoint:
        tokens = checkpoint.get("tokens_collected", 0)
        chunk = checkpoint.get("last_chunk_id", 0)
        target = metadata.get("tokens_per_shard") if metadata else None
        if target:
            pct = tokens / target * 100
            return (
                f"Shard {shard_id}: {format_int(tokens)} / {format_int(target)} tokens "
                f"({pct:.1f}%) - Chunk {chunk}"
            )
        return f"Shard {shard_id}: {format_int(tokens)} tokens - Chunk {chunk}"

    if metadata:
        return f"Shard {shard_id}: Starting up..."

    return f"Shard {shard_id}: Not started"


def human_readable_size(bytes_count: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(bytes_count)
    for suffix in suffixes:
        if size < 1024.0:
            return f"{size:,.2f} {suffix}"
        size /= 1024.0
    return f"{size:.2f} EB"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def disk_usage_summary(num_shards: int) -> Tuple[List[str], str]:
    summaries = []
    for shard in range(num_shards):
        shard_path = REPO_ROOT / "data" / f"shard{shard}"
        size_bytes = directory_size(shard_path)
        summaries.append(f"{shard_path}: {human_readable_size(size_bytes)}")

    total_bytes = directory_size(REPO_ROOT / "data")
    return summaries, human_readable_size(total_bytes)


def gpu_status() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "nvidia-smi not found"

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = []
        for raw in result.stdout.strip().splitlines():
            idx, used, total, util = [item.strip() for item in raw.split(",")]
            lines.append(f"  GPU {idx}: {used} / {total} MB ({util}% util)")
        return "\n".join(lines) if lines else "No GPU data"
    except subprocess.CalledProcessError:
        return "Failed to query GPU stats"


def disk_space_line() -> str:
    usage = shutil.disk_usage(REPO_ROOT)
    used = human_readable_size(usage.used)
    total = human_readable_size(usage.total)
    percent = usage.used / usage.total * 100
    return f"  Used: {used} / {total} ({percent:.1f}% full)"


def tail_errors(log_path: Path, limit: int = 2) -> List[str]:
    if not log_path.exists():
        return []

    keywords = ("error", "exception", "warning")
    matches: deque[str] = deque(maxlen=limit)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                lower = line.lower()
                if any(keyword in lower for keyword in keywords):
                    matches.append(line.rstrip())
    except OSError:
        return []
    return list(matches)


def render_monitor(args: argparse.Namespace) -> None:
    clear_screen()
    line = "=" * 40
    print(line)
    print("Kimi-K2 Activation Collection Monitor")
    print(line)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("")

    print("Shard Progress:")
    print("----------------------------------------")
    for shard in range(args.num_shards):
        print(shard_status(shard))

    print("\nDisk Usage:")
    print("----------------------------------------")
    shard_summaries, total_summary = disk_usage_summary(args.num_shards)
    for summary in shard_summaries:
        print(f"  {summary}")
    print(f"\nTotal data/: {total_summary}")

    print("\nSystem Resources:")
    print("----------------------------------------")
    print("GPU Memory Usage:")
    print(gpu_status())
    print("\nDisk Space:")
    print(disk_space_line())

    print("\nRecent Errors (last 2 per shard):")
    print("----------------------------------------")
    any_errors = False
    for shard in range(args.num_shards):
        log_path = REPO_ROOT / "logs" / f"shard{shard}.log"
        errors = tail_errors(log_path, limit=2)
        if errors:
            any_errors = True
            print(f"Shard {shard}:")
            for err in errors:
                print(f"  {err}")

    if not any_errors:
        print("  No errors detected")

    print("\n" + line)
    print("Press Ctrl+C to stop. Refreshing in "
          f"{args.refresh_interval} seconds...")
    print(line)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor activation collection progress.")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shard directories to monitor.")
    parser.add_argument("--refresh-interval", type=float, default=10.0, help="Refresh interval in seconds.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    try:
        while True:
            render_monitor(args)
            time.sleep(args.refresh_interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
