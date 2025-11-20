#!/usr/bin/env python3
"""
Full activation collection launcher implemented in Python.

Replaces the original Bash script, launching four shard processes that
collect 1B tokens in total (250M tokens per shard by default).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def print_header(title: str) -> None:
    line = "=" * 40
    print(f"{line}\n{title}\n{line}\n")


def confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def ensure_directories(num_shards: int) -> None:
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    for shard in range(num_shards):
        (REPO_ROOT / "data" / f"shard{shard}" / "activations").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "data" / f"shard{shard}" / "token_ids").mkdir(parents=True, exist_ok=True)


def available_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


@dataclass
class ShardProcess:
    shard_id: int
    process: subprocess.Popen
    log_path: Path


def launch_shard(
    shard_id: int,
    gpu_list: str,
    args: argparse.Namespace,
) -> ShardProcess:
    log_path = REPO_ROOT / "logs" / f"shard{shard_id}.log"
    command = [
        sys.executable,
        "core/collect_activations.py",
        "--shard_id",
        str(shard_id),
        "--num_shards",
        str(args.num_shards),
        "--output_dir",
        f"data/shard{shard_id}",
        "--target_layer",
        str(args.target_layer),
        "--tokens_per_shard",
        str(args.tokens_per_shard),
        "--chunk_size",
        str(args.chunk_size),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_list

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    return ShardProcess(shard_id=shard_id, process=process, log_path=log_path)


def run_collection(args: argparse.Namespace) -> int:
    ensure_directories(args.num_shards)
    print_header("Kimi-K2 Full Collection (1B tokens)")
    print(f"Launching {args.num_shards} parallel shards\n")

    if not confirm(
        "This run may require ~3-4 hours and approximately 15 TB of disk space. "
        "Continue? (y/n) "
    ):
        print("Cancelled by user.")
        return 0

    available_gb = available_disk_gb(REPO_ROOT)
    if available_gb < args.required_disk_gb:
        print(
            f"[warn] Only {available_gb:,.0f} GB available. "
            f"Recommended at least {args.required_disk_gb:,.0f} GB."
        )
        if not confirm("Continue anyway? (y/n) "):
            return 1

    shard_envs: List[str] = args.cuda_groups or []
    if shard_envs and len(shard_envs) != args.num_shards:
        raise ValueError("Number of CUDA groups must match num_shards.")
    if not shard_envs:
        shard_envs = [f"{2*i},{2*i+1}" for i in range(args.num_shards)]

    shards: List[ShardProcess] = []
    for shard_id, gpu_list in enumerate(shard_envs):
        print(f"Starting shard {shard_id} on GPUs {gpu_list}...")
        shard = launch_shard(shard_id, gpu_list, args)
        shards.append(shard)
        print(f"  PID: {shard.process.pid}")
        time.sleep(args.launch_delay)

    print("\nAll shards launched. Logs are stored in logs/shard*.log\n")
    print("Monitor progress with: python scripts/monitor.py")
    print("Waiting for shard processes to finish...\n")

    exit_code = 0
    try:
        for shard in shards:
            code = shard.process.wait()
            if code != 0:
                print(
                    f"[error] Shard {shard.shard_id} exited with status {code}. "
                    f"See {shard.log_path}."
                )
                exit_code = code or exit_code
    except KeyboardInterrupt:
        print("\n[warn] Keyboard interrupt detected. Stopping shard processes...")
        for shard in shards:
            shard.process.terminate()
        raise

    print("\nAll shard processes completed.")
    print("Verifying collected data...\n")

    verify_command = [
        sys.executable,
        "core/verify_collection.py",
    ]
    for shard_id in range(args.num_shards):
        verify_command.extend(["--data_dir", f"data/shard{shard_id}"])

    verify_code = subprocess.call(verify_command, cwd=REPO_ROOT)
    if verify_code == 0:
        print("\nCollection complete and verified!")
        print(
            "Next step: train SAE\n"
            "  python core/train_sae.py "
            "--data_dir data --output_dir models/sae_layer45_8x "
            "--num_shards 4 --target_layer 45 "
            "--d_model 7168 --d_sae 57344 --l1_coeff 1e-3 --lr 3e-4 "
            "--batch_size 8192 --epochs 3"
        )
    else:
        print("\n[error] Verification failed. Inspect shard logs for details.")

    return exit_code or verify_code


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch full activation collection across shards.")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shard processes to launch.")
    parser.add_argument("--tokens-per-shard", type=int, default=250_000_000, help="Tokens per shard.")
    parser.add_argument("--chunk-size", type=int, default=10_000_000, help="Tokens per saved chunk.")
    parser.add_argument("--target-layer", type=int, default=45, help="Transformer layer to probe.")
    parser.add_argument(
        "--cuda-groups",
        nargs="*",
        default=None,
        help="Space-separated list of CUDA device groups per shard (e.g. '0,1 2,3').",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between shard launches.",
    )
    parser.add_argument(
        "--required-disk-gb",
        type=int,
        default=15_000,
        help="Expected disk requirement that triggers a warning if not met.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    exit_code = run_collection(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
