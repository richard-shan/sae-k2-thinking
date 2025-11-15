#!/bin/bash

# Monitor collection progress across all shards

while true; do
    clear
    echo "========================================="
    echo "Kimi-K2 Activation Collection Monitor"
    echo "========================================="
    date
    echo ""
    
    # Check each shard's progress
    echo "Shard Progress:"
    echo "----------------------------------------"
    
    for shard in 0 1 2 3; do
        CHECKPOINT="data/shard${shard}/checkpoint.json"
        METADATA="data/shard${shard}/metadata.json"
        
        if [ -f "$CHECKPOINT" ]; then
            # Extract metrics using python
            TOKENS=$(python3 -c "import json; print(json.load(open('$CHECKPOINT'))['tokens_collected'])" 2>/dev/null || echo "0")
            CHUNKS=$(python3 -c "import json; print(json.load(open('$CHECKPOINT'))['last_chunk_id'])" 2>/dev/null || echo "0")
            
            if [ -f "$METADATA" ]; then
                TARGET=$(python3 -c "import json; print(json.load(open('$METADATA'))['tokens_per_shard'])" 2>/dev/null || echo "250000000")
                PCT=$(python3 -c "print(f'{($TOKENS / $TARGET * 100):.1f}')" 2>/dev/null || echo "0.0")
                
                # Format numbers with commas
                TOKENS_FMT=$(printf "%'d" $TOKENS 2>/dev/null || echo $TOKENS)
                TARGET_FMT=$(printf "%'d" $TARGET 2>/dev/null || echo $TARGET)
                
                echo "Shard $shard: $TOKENS_FMT / $TARGET_FMT tokens ($PCT%) - Chunk $CHUNKS"
            else
                echo "Shard $shard: $TOKENS tokens - Chunk $CHUNKS"
            fi
        elif [ -f "$METADATA" ]; then
            echo "Shard $shard: Starting up..."
        else
            echo "Shard $shard: Not started"
        fi
    done
    
    echo ""
    echo "Disk Usage:"
    echo "----------------------------------------"
    du -sh data/shard* 2>/dev/null | sort || echo "No data yet"
    
    echo ""
    TOTAL=$(du -sh data/ 2>/dev/null | cut -f1 || echo "0")
    echo "Total: $TOTAL"
    
    echo ""
    echo "System Resources:"
    echo "----------------------------------------"
    
    # GPU utilization
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Usage:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r idx used total util; do
            printf "  GPU %s: %5s / %5s MB (%2s%% util)\n" "$idx" "$used" "$total" "$util"
        done
    fi
    
    echo ""
    
    # Disk space
    echo "Disk Space:"
    df -h . | tail -1 | awk '{printf "  Used: %s / %s (%s full)\n", $3, $2, $5}'
    
    echo ""
    echo "Recent Errors (last 2 per shard):"
    echo "----------------------------------------"
    
    for shard in 0 1 2 3; do
        LOGFILE="logs/shard${shard}.log"
        if [ -f "$LOGFILE" ]; then
            ERRORS=$(grep -i "error\|exception\|warning" "$LOGFILE" | tail -2)
            if [ -n "$ERRORS" ]; then
                echo "Shard $shard:"
                echo "$ERRORS" | sed 's/^/  /'
            fi
        fi
    done
    
    # If no errors found
    if ! grep -qi "error\|exception\|warning" logs/shard*.log 2>/dev/null; then
        echo "  No errors detected"
    fi
    
    echo ""
    echo "========================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    echo "========================================="
    
    sleep 10
done
