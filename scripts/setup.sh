#!/bin/bash

echo "========================================="
echo "Kimi-K2 SAE Project Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 9 ]; then
    echo "✗ Python 3.9+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# Check CUDA
echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "✓ CUDA $CUDA_VERSION"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "✗ nvidia-smi not found - GPU required"
    exit 1
fi

# Check disk space
echo ""
echo "Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available: ${AVAILABLE} GB"
if [ "$AVAILABLE" -lt 15000 ]; then
    echo "⚠ Warning: Less than 15 TB available"
    echo "  Full collection requires ~15 TB"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "✗ Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/{shard0,shard1,shard2,shard3}/{activations,token_ids}
mkdir -p logs
mkdir -p models
echo "✓ Directories created"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x launch_pilot.sh
chmod +x launch_collection.sh
chmod +x monitor.sh
echo "✓ Scripts are executable"

# Test imports
echo ""
echo "Testing imports..."
python3 -c "import torch; import transformers; import datasets; print('✓ All imports successful')"

if [ $? -ne 0 ]; then
    echo "✗ Import test failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run pilot: bash launch_pilot.sh"
echo "  2. If successful, run full collection: bash launch_collection.sh"
echo "  3. Monitor progress: bash monitor.sh"
echo ""
echo "For detailed usage, see USAGE.md"
echo "========================================="
