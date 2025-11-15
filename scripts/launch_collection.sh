#!/bin/bash
set -e

echo "========================================="
echo "Kimi-K2 Full Collection (1B tokens)"
echo "Launching 4 parallel shards"
echo "========================================="
echo ""

# Confirm with user
read -p "This will run for ~3-4 hours and use ~15 TB of disk space. Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create directories
mkdir -p data/{shard0,shard1,shard2,shard3}/{activations,token_ids}
mkdir -p logs

# Check disk space
echo "Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED=15000

if [ "$AVAILABLE" -lt "$REQUIRED" ]; then
    echo "⚠ Warning: May not have enough disk space"
    echo "  Available: ${AVAILABLE} GB"
    echo "  Required: ~15 TB (15,000 GB)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo ""
echo "========================================="
echo "Launching Collection Processes"
echo "========================================="
echo ""

# Function to check if a process is still running
check_process() {
    if ps -p $1 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Launch shard 0 (GPUs 0-1)
echo "Starting Shard 0 (GPUs 0-1)..."
CUDA_VISIBLE_DEVICES=0,1 python scripts/collect_activations.py \
    --shard_id 0 \
    --num_shards 4 \
    --output_dir data/shard0 \
    --target_layer 45 \
    --tokens_per_shard 250000000 \
    --chunk_size 10000000 \
    > logs/shard0.log 2>&1 &
PID0=$!
echo "  PID: $PID0"

# Small delay to avoid simultaneous model downloads
sleep 5

# Launch shard 1 (GPUs 2-3)
echo "Starting Shard 1 (GPUs 2-3)..."
CUDA_VISIBLE_DEVICES=2,3 python scripts/collect_activations.py \
    --shard_id 1 \
    --num_shards 4 \
    --output_dir data/shard1 \
    --target_layer 45 \
    --tokens_per_shard 250000000 \
    --chunk_size 10000000 \
    > logs/shard1.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

sleep 5

# Launch shard 2 (GPUs 4-5)
echo "Starting Shard 2 (GPUs 4-5)..."
CUDA_VISIBLE_DEVICES=4,5 python scripts/collect_activations.py \
    --shard_id 2 \
    --num_shards 4 \
    --output_dir data/shard2 \
    --target_layer 45 \
    --tokens_per_shard 250000000 \
    --chunk_size 10000000 \
    > logs/shard2.log 2>&1 &
PID2=$!
echo "  PID: $PID2"

sleep 5

# Launch shard 3 (GPUs 6-7)
echo "Starting Shard 3 (GPUs 6-7)..."
CUDA_VISIBLE_DEVICES=6,7 python scripts/collect_activations.py \
    --shard_id 3 \
    --num_shards 4 \
    --output_dir data/shard3 \
    --target_layer 45 \
    --tokens_per_shard 250000000 \
    --chunk_size 10000000 \
    > logs/shard3.log 2>&1 &
PID3=$!
echo "  PID: $PID3"

echo ""
echo "========================================="
echo "All shards launched!"
echo "========================================="
echo ""
echo "Process IDs:"
echo "  Shard 0: $PID0"
echo "  Shard 1: $PID1"
echo "  Shard 2: $PID2"
echo "  Shard 3: $PID3"
echo ""
echo "Monitor progress:"
echo "  bash monitor.sh"
echo ""
echo "View logs:"
echo "  tail -f logs/shard0.log"
echo "  tail -f logs/shard1.log"
echo "  tail -f logs/shard2.log"
echo "  tail -f logs/shard3.log"
echo ""
echo "Waiting for all processes to complete..."
echo "This will take approximately 3-4 hours."
echo ""

# Wait for all processes
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "========================================="
echo "All Shards Complete!"
echo "========================================="
echo ""

# Verify all shards
echo "Verifying collected data..."
python scripts/verify_collection.py \
    --data_dir data/shard0 \
    --data_dir data/shard1 \
    --data_dir data/shard2 \
    --data_dir data/shard3

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Collection Complete and Verified!"
    echo "========================================="
    echo ""
    echo "Next step: Train SAE"
    echo "  python scripts/train_sae.py \\"
    echo "    --data_dir data \\"
    echo "    --output_dir models/sae_layer45_8x \\"
    echo "    --num_shards 4 \\"
    echo "    --target_layer 45 \\"
    echo "    --d_model 7168 \\"
    echo "    --d_sae 57344 \\"
    echo "    --l1_coeff 1e-3 \\"
    echo "    --lr 3e-4 \\"
    echo "    --batch_size 8192 \\"
    echo "    --epochs 3"
else
    echo ""
    echo "========================================="
    echo "✗ Verification Failed"
    echo "========================================="
    echo ""
    echo "Please check logs for errors:"
    echo "  logs/shard*.log"
fi

echo ""
echo "Collection took approximately $(date -u -d @$SECONDS +%T)"
echo "========================================="
