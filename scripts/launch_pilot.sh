#!/bin/bash
set -e

echo "========================================="
echo "Kimi-K2 Pilot Collection (10M tokens)"
echo "========================================="
echo ""

# Create pilot directories
mkdir -p data/pilot/{activations,token_ids}
mkdir -p logs

echo "Running pilot collection on 10M tokens..."
echo "This will take approximately 5-10 minutes."
echo ""

# Launch single-stream pilot on GPUs 0-1
CUDA_VISIBLE_DEVICES=0,1 python core/collect_activations.py \
    --shard_id 0 \
    --num_shards 1 \
    --output_dir data/pilot \
    --target_layer 45 \
    --tokens_per_shard 10000000 \
    --chunk_size 5000000 \
    2>&1 | tee logs/pilot.log

echo ""
echo "========================================="
echo "Pilot collection complete!"
echo "========================================="
echo ""

# Verify the pilot data
echo "Verifying pilot data..."
python core/verify_collection.py --data_dir data/pilot

echo ""
echo "========================================="
echo "Pilot Results"
echo "========================================="
echo ""
echo "Logs saved to: logs/pilot.log"
echo "Data saved to: data/pilot/"
echo ""

# Check if verification passed
if [ $? -eq 0 ]; then
    echo "✓ Pilot successful!"
    echo ""
    echo "Next steps:"
    echo "  1. Review logs/pilot.log for any warnings"
    echo "  2. Check disk space: df -h"
    echo "  3. Run full collection: bash launch_collection.sh"
else
    echo "✗ Pilot failed verification"
    echo ""
    echo "Please check logs/pilot.log for errors before proceeding."
fi

echo "========================================="
