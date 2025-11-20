#!/usr/bin/env python3
"""
Pilot activation collection launcher implemented in Python.

Replaces the original Bash script so the pipeline can be run without
shell access. The script launches a single pilot shard, streams logs to
stdout, and verifies the collected data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def print_header(title: str) -> None:
    line = "=" * 40
    print(f"{line}\n{title}\n{line}\n")


def ensure_directories() -> None:
    (REPO_ROOT / "data" / "pilot" / "activations").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "pilot" / "token_ids").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def stream_process(command: Iterable[str], log_path: Path, env: Optional[dict] = None) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
        except KeyboardInterrupt:
            print("\n[warn] Pilot interrupted by user. Stopping subprocess...")
            process.terminate()
        finally:
            process.wait()

        return process.returncode


def run_pilot(args: argparse.Namespace) -> int:
    ensure_directories()
    log_path = REPO_ROOT / "logs" / "pilot.log"

    print_header("Kimi-K2 Pilot Collection (10M tokens)")
    print("Running pilot collection...")
    print("This run typically finishes within 5-10 minutes.\n")

    env = os.environ.copy()
    if args.cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    command = [
        sys.executable,
        "core/collect_activations.py",
        "--shard_id",
        "0",
        "--num_shards",
        "1",
        "--output_dir",
        "data/pilot",
        "--target_layer",
        str(args.target_layer),
        "--tokens_per_shard",
        str(args.tokens),
        "--chunk_size",
        str(args.chunk_size),
    ]

    return_code = stream_process(command, log_path, env=env)

    print("\nPilot run finished.\n")

    if return_code != 0:
        print(f"[error] Pilot collection exited with status {return_code}. "
              f"See {log_path} for details.")
        return return_code

    print("Verifying pilot data...")
    verify_command = [
        sys.executable,
        "core/verify_collection.py",
        "--data_dir",
        "data/pilot",
    ]

    verify_code = subprocess.call(verify_command, cwd=REPO_ROOT)
    if verify_code == 0:
        print("\nPilot successful!")
        print(f"Logs saved to: {log_path}")
        print("Data saved to: data/pilot/\n")
        print("Next steps:")
        print("  1. Review logs/pilot.log for warnings")
        print("  2. Confirm sufficient disk space (recommend df -h)")
        print("  3. Launch full collection: python scripts/launch_collection.py")
    else:
        print("\n[error] Pilot verification failed. "
              f"Inspect {log_path} and rerun once issues are resolved.")

    return verify_code


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch pilot activation collection (10M tokens).")
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="CUDA device list for the pilot shard (default: 0,1,2,3,4,5,6,7).",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=10_000_000,
        help="Number of tokens to collect in the pilot run.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Number of tokens per saved chunk.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=45,
        help="Transformer layer to probe for activations.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    exit_code = run_pilot(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
