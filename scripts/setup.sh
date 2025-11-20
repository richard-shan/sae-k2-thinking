#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "========================================="
echo "Kimi-K2 SAE Project Setup"
echo "========================================="
echo ""

echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)

if [ "${PYTHON_MAJOR}" -lt 3 ] || { [ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 9 ]; }; then
    echo "[error] Python 3.9+ required (found ${PYTHON_VERSION})"
    exit 1
fi
echo "[ok] Python ${PYTHON_VERSION}"

echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version/ {print $9; exit}')
    echo "[ok] CUDA ${CUDA_VERSION}"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "[error] nvidia-smi not found. GPUs are required."
    exit 1
fi

echo ""
echo "Checking disk space..."
AVAILABLE=$(df -BG "${REPO_ROOT}" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available: ${AVAILABLE} GB"
if [ "${AVAILABLE}" -lt 15000 ]; then
    echo "[warn] Less than 15 TB available. Full collection requires ~15 TB."
fi

echo ""
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt
echo "[ok] Dependencies installed"

echo ""
echo "Creating project directories..."
mkdir -p data/{shard0,shard1,shard2,shard3}/{activations,token_ids}
mkdir -p logs
mkdir -p models
echo "[ok] Directory layout ready"

echo ""
echo "Ensuring shell scripts are executable..."
chmod +x scripts/launch_pilot.sh
chmod +x scripts/launch_collection.sh
chmod +x scripts/monitor.sh
echo "[ok] Shell scripts marked executable"

echo ""
echo "Testing core Python imports..."
python3 -c "import torch, transformers, datasets; print('[ok] Core imports succeeded')"

echo ""
echo "========================================="
echo "[ok] Setup Complete!"
echo "========================================="
echo "Next steps:"
echo "  1. Run pilot: bash launch_pilot.sh"
echo "  2. If successful, run full collection: bash launch_collection.sh"
echo "  3. Monitor progress: bash monitor.sh"
echo ""
echo "For detailed usage, see docs/USAGE.md"
echo "========================================="
