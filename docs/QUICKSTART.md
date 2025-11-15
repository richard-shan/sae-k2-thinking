# Quick Start Cheatsheet

## Copy-Paste Commands

### Initial Setup (5 minutes)
```bash
git clone https://github.com/yourusername/kimi-sae-project.git
cd kimi-sae-project
bash setup.sh
```

### Phase 1: Collection (~4 hours, ~$100)

**Pilot Test (10 min)**
```bash
bash launch_pilot.sh
```

**Full Collection (4 hours)**
```bash
# Terminal 1: Launch collection
bash launch_collection.sh

# Terminal 2: Monitor progress
bash monitor.sh
```

### Phase 2: Training (~1 hour, ~$5-24)
```bash
python scripts/train_sae.py \
    --data_dir data \
    --output_dir models/sae_layer45_8x \
    --num_shards 4 \
    --target_layer 45 \
    --d_model 7168 \
    --d_sae 57344 \
    --l1_coeff 1e-3 \
    --lr 3e-4 \
    --batch_size 8192 \
    --epochs 3
```

### Phase 3: Analysis (local)

**Feature Statistics**
```bash
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --statistics
```

**Analyze Specific Feature**
```bash
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --feature_id 12345 \
    --top_k 20
```

## File Locations

| What | Where |
|------|-------|
| Collected activations | `data/shard0-3/activations/` |
| Token IDs | `data/shard0-3/token_ids/` |
| Logs | `logs/shard*.log` |
| Trained SAE | `models/sae_layer45_8x/sae_best.pt` |
| Checkpoints | `data/shard*/checkpoint.json` |

## Common Commands

**Check GPU Usage**
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

**Check Disk Space**
```bash
df -h
du -sh data/shard*
```

**View Logs**
```bash
tail -f logs/shard0.log
tail -100 logs/shard0.log | grep -i error
```

**Kill All Collection Processes**
```bash
pkill -f collect_activations.py
```

**Resume Collection After Crash**
```bash
# Just re-run - will resume from checkpoint
bash launch_collection.sh
```

**Verify Data**
```bash
python scripts/verify_collection.py \
    --data_dir data/shard0 \
    --data_dir data/shard1 \
    --data_dir data/shard2 \
    --data_dir data/shard3
```

## Parameter Tuning

### Adjust Sparsity
```bash
# More sparse (L0 ~ 30-50)
python scripts/train_sae.py ... --l1_coeff 3e-3

# Less sparse (L0 ~ 100-150)  
python scripts/train_sae.py ... --l1_coeff 3e-4
```

### Different Expansion Factors
```bash
# 4× expansion (28k features)
python scripts/train_sae.py ... --d_sae 28672

# 8× expansion (57k features) - default
python scripts/train_sae.py ... --d_sae 57344

# 16× expansion (114k features)
python scripts/train_sae.py ... --d_sae 114688
```

### Different Layers
```bash
# Modify target_layer in launch_collection.sh
# Then collect from different layers:
bash launch_collection.sh  # --target_layer 30
bash launch_collection.sh  # --target_layer 45
bash launch_collection.sh  # --target_layer 55
```

## Troubleshooting One-Liners

**CUDA OOM during collection**
```bash
# Edit collect_activations.py, reduce batch_size to 1
```

**Training too slow**
```bash
python scripts/train_sae.py ... --batch_size 16384 --num_workers 2
```

**Disk full**
```bash
# Free space, collection will resume from checkpoint
rm -rf data/pilot
```

**Model download timeout**
```bash
huggingface-cli download moonshotai/Kimi-K2-Thinking
```

**Check if processes still running**
```bash
ps aux | grep collect_activations
```

## Expected Outputs

### Pilot Success
```
Shard 0: 10,000,000 tokens collected
✓ All checks passed!
```

### Collection Progress
```
Shard 0: 87,432,100 / 250,000,000 tokens (35.0%)
Shard 1: 92,145,890 / 250,000,000 tokens (36.9%)
```

### Training Metrics
```
Epoch 1:
  Train - Loss: 0.007123, L0: 65.4
  Val   - Loss: 0.007089, L0: 64.8
  ✓ New best model!
```

## Timeline

| Task | Time | Cost |
|------|------|------|
| Setup | 5 min | $0 |
| Pilot | 10 min | $4 |
| Collection | 4 hrs | $96 |
| Training | 1 hr | $5-24 |
| **Total** | **~5 hrs** | **~$105** |

## URLs & Resources

- **Model**: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- **Dataset**: https://huggingface.co/datasets/tiiuae/falcon-refinedweb
- **Lambda Labs**: https://lambdalabs.com/
- **Documentation**: See README.md, USAGE.md, PROJECT_SUMMARY.md

## Emergency Contacts

**If something goes wrong:**
1. Check `logs/shard*.log`
2. Check `monitor.sh` output
3. Verify data: `python scripts/verify_collection.py ...`
4. Open GitHub issue
5. Contact maintainer

## Success Checklist

- [ ] Setup completed successfully
- [ ] Pilot test passed (10M tokens)
- [ ] Full collection completed (1B tokens)
- [ ] Data verification passed
- [ ] SAE training completed
- [ ] Model saved and metadata generated
- [ ] Can load and use trained SAE

---

**You're ready to go!** Start with `bash setup.sh` then `bash launch_pilot.sh`.

For detailed explanations, see USAGE.md.
