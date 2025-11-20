#!/usr/bin/env python3
"""
Project setup utility for the Kimi-K2 SAE pipeline.

Replaces the original Bash script so the setup process can be run on
systems where shell scripts are not available or permitted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def print_header(title: str) -> None:
    line = "=" * 40
    print(f"{line}\n{title}\n{line}\n")


def check_python_version() -> None:
    print("Checking Python version...")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 9):
        raise RuntimeError(
            f"Python 3.9+ required (found {version_str}). "
            "Please upgrade before continuing."
        )
    print(f"[ok] Python {version_str}\n")


def check_cuda() -> None:
    print("Checking CUDA availability...")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi not found. GPUs are required for this pipeline.")

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version,cuda_version,name,memory.total", "--format=csv"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to query GPU info: {exc}") from exc

    print("[ok] CUDA device information:\n")
    print(result.stdout.strip())
    print("")


def check_disk_space() -> None:
    print("Checking disk space...")
    usage = shutil.disk_usage(REPO_ROOT)
    available_gb = usage.free / (1024 ** 3)
    print(f"Available: {available_gb:,.0f} GB")
    if available_gb < 15_000:
        print("[warn] Less than 15 TB available. "
              "Full collection requires approximately 15 TB.\n")
    else:
        print("")


def install_dependencies() -> None:
    print("Installing Python dependencies...")
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    print("[ok] Dependencies installed\n")


def create_directories() -> None:
    print("Creating project directories...")
    dirs = [
        REPO_ROOT / "data" / f"shard{i}" / "activations"
        for i in range(4)
    ]
    dirs += [
        REPO_ROOT / "data" / f"shard{i}" / "token_ids"
        for i in range(4)
    ]
    dirs += [
        REPO_ROOT / "logs",
        REPO_ROOT / "models",
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print("[ok] Directory layout ready\n")


def test_core_imports() -> None:
    print("Testing core Python imports...")
    test_code = (
        "import torch, transformers, datasets;"
        "print('[ok] Core imports succeeded')"
    )
    subprocess.run(
        [sys.executable, "-c", test_code],
        check=True,
    )
    print("")


def main() -> None:
    os.chdir(REPO_ROOT)

    print_header("Kimi-K2 SAE Project Setup")

    try:
        check_python_version()
        check_cuda()
        check_disk_space()
        install_dependencies()
        create_directories()
        test_core_imports()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] Setup failed: {exc}")
        sys.exit(1)

    print_header("[ok] Setup Complete!")
    print("Next steps:")
    print("  1. Run pilot: python scripts/launch_pilot.py")
    print("  2. If successful, run full collection: python scripts/launch_collection.py")
    print("  3. Monitor progress: python scripts/monitor.py")
    print("\nFor detailed usage, see docs/USAGE.md")


if __name__ == "__main__":
    main()
