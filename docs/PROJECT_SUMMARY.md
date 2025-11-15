# Kimi-K2 SAE Project Summary

## Quick Reference

### Project Structure
```
kimi-sae-project/
├── README.md              # Main documentation
├── USAGE.md              # Detailed usage guide
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── LICENSE              # MIT License
│
├── scripts/
│   ├── collect_activations.py    # Activation collection (Phase 1)
│   ├── train_sae.py              # SAE training (Phase 2)
│   ├── verify_collection.py      # Data verification
│   └── interpret_features.py     # Feature analysis (Phase 3)
│
├── utils/
│   ├── __init__.py
│   ├── sae_model.py             # SAE architecture
│   └── dataset.py               # Data loading utilities
│
├── examples/
│   └── analyze_reasoning.py     # Example reasoning analysis
│
└── launch scripts:
    ├── launch_pilot.sh          # Quick 10M token test
    ├── launch_collection.sh     # Full 1B token collection
    └── monitor.sh              # Real-time progress monitoring
```

## Complete Workflow

### Phase 1: Collection (~4 hours, $96)
```bash
# 1. Setup
bash setup.sh

# 2. Pilot test (5-10 min)
bash launch_pilot.sh

# 3. Full collection (3-4 hours) 
bash launch_collection.sh

# 4. Monitor (in separate terminal)
bash monitor.sh
```

**Output**: 14.3 TB of activation data in `data/shard0-3/`

### Phase 2: Training (~1 hour, $5-24)
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

**Output**: Trained SAE in `models/sae_layer45_8x/sae_best.pt` (~3 GB)

### Phase 3: Interpretation (local)
```bash
# Get feature statistics
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --statistics

# Analyze specific feature
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --feature_id 12345 \
    --top_k 20
```

## Key Files Explained

### Collection Script (`collect_activations.py`)
- Loads Kimi-K2-Thinking via HuggingFace Transformers
- Registers PyTorch hook on target layer
- Streams text corpus (RefinedWeb by default)
- Extracts activation vectors during forward pass
- Saves continuously (every 10M tokens) for fault tolerance
- Supports checkpoint recovery

**Key parameters**:
- `--shard_id`: Which parallel shard (0-3)
- `--target_layer`: Which K2 layer to extract (default: 45)
- `--tokens_per_shard`: Tokens to collect (default: 250M)
- `--chunk_size`: Save frequency (default: 10M)

### Training Script (`train_sae.py`)
- Loads activations via memory-mapped dataset (doesn't load 14 TB into RAM)
- Trains sparse autoencoder with L1 penalty
- Normalizes decoder columns after each step
- Saves best model by validation loss
- Tracks L0 sparsity and reconstruction quality

**Key parameters**:
- `--d_sae`: SAE hidden dimension (default: 57344 = 8× expansion)
- `--l1_coeff`: Sparsity penalty (default: 1e-3)
- `--lr`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 8192)

### SAE Model (`utils/sae_model.py`)
```python
class SparseAutoencoder:
    encoder: Linear(7168 → 57344)
    decoder: Linear(57344 → 7168)
    
    forward(x):
        features = ReLU(encoder(x))      # Sparsity via ReLU
        reconstructed = decoder(features)
        return reconstructed, features
```

**Design choices**:
- ReLU activation for natural sparsity
- Unit-norm decoder columns for interpretability
- Bias terms in both encoder and decoder

### Dataset Loader (`utils/dataset.py`)
- Memory-mapped loading: Doesn't load 14 TB into RAM
- Streams from disk during training
- Supports multiple shards automatically
- Random train/val split (98/2 by default)

## Hardware Requirements

### Phase 1: Collection
- **Minimum**: 4× A100 (80GB)
- **Recommended**: 8× H100 (80GB)
- **Storage**: 15 TB SSD
- **Time**: 3-4 hours (parallel) or 10-12 hours (single-stream)

### Phase 2: Training
- **Minimum**: 1× A100 (40GB)
- **Recommended**: 1× H100 (80GB)
- **Storage**: 14.3 TB (same data from Phase 1)
- **Time**: 1 hour

### Phase 3: Interpretation
- **Minimum**: 1× A100 (40GB) or local GPU
- **Storage**: Only need trained SAE (~3 GB)
- **Time**: Seconds to minutes per query

## Cost Breakdown

### Lambda Labs (8× H100 @ $23.92/hr)
| Phase | Time | Cost |
|-------|------|------|
| Setup | 15 min | $6 |
| Pilot | 10 min | $4 |
| Collection | 4 hrs | $96 |
| Training | 1 hr | $24 |
| **Total** | **~5.5 hrs** | **~$130** |

### Optimized (separate instances)
| Phase | Instance | Time | Cost |
|-------|----------|------|------|
| Collection | 8× H100 | 4 hrs | $96 |
| Training | 1× H100 | 1 hr | $5 |
| **Total** | | **5 hrs** | **$101** |

## Data Specifications

### Activations
- **Format**: NumPy `.npy` (FP16)
- **Shape**: `[n_tokens, 7168]`
- **Size**: 14.3 GB per 1M tokens
- **Total**: 14.3 TB for 1B tokens

### Token IDs
- **Format**: NumPy `.npy` (int32)
- **Shape**: `[n_tokens]`
- **Size**: 4 MB per 1M tokens
- **Total**: 4 GB for 1B tokens

### Trained SAE
- **Format**: PyTorch checkpoint `.pt`
- **Components**: encoder, decoder, biases
- **Size**: ~3 GB (for 8× expansion)
- **Includes**: Training metadata, hyperparameters, metrics

## Troubleshooting Quick Reference

### "CUDA out of memory"
→ Reduce batch size: `--batch_size 1` in collection
→ Use 4-bit quantization (slower): add `load_in_4bit=True`

### "No space left on device"
→ Check space: `df -h`
→ Collection saves checkpoints, can resume after freeing space

### "Model download timeout"
→ Pre-download: `huggingface-cli download moonshotai/Kimi-K2-Thinking`

### "DataLoader worker crashed"
→ Reduce workers: `--num_workers 0` in training

### "Collection too slow"
→ Use parallel collection (4 streams on 8 GPUs)
→ Check GPU utilization: `nvidia-smi`

### "Training converges poorly"
→ Adjust L1 coefficient (target L0: 50-100)
→ Train longer: `--epochs 5`
→ Verify data with `verify_collection.py`

## Research Applications

### 1. Identifying Reasoning Features
Which features activate during:
- Mathematical calculation
- Logical deduction  
- Chain-of-thought generation
- Planning and problem decomposition

### 2. Studying Reasoning Circuits
How do features compose during:
- Multi-step reasoning
- Error correction
- Explanation generation

### 3. Analyzing Complexity Scaling
How do activations change from:
- Simple problems (2+2)
- Medium problems (algebra)
- Complex problems (multi-step proofs)

### 4. Causal Interventions
Ablate specific features and measure:
- Impact on accuracy
- Changes in reasoning approach
- Error patterns

### 5. Cross-Model Comparison
Compare features across:
- Different layers (30, 45, 55)
- Different models (K2 vs GPT-4 vs Claude)
- Different training regimes

## Best Practices

### For Reproducibility
- Save all metadata (corpus, hyperparameters, seeds)
- Use fixed random seeds
- Document exact commands used
- Version control code

### For Efficiency  
- Run pilot first (catches issues early)
- Use parallel collection when possible
- Separate expensive and cheap phases
- Use spot/preemptible instances

### For Research Quality
- Validate on held-out data
- Test multiple hyperparameters
- Compare against baselines
- Document limitations

## Next Steps

After completing this pipeline:

1. **Baseline Analysis**: Establish performance metrics
2. **Feature Discovery**: Identify reasoning-specific features
3. **Ablation Studies**: Test causal role of features
4. **Cross-Dataset Evaluation**: Test on GSM8K, MATH, GPQA
5. **Comparison**: Train SAEs on other layers/models
6. **Publication**: Document findings in paper

## Citation

```bibtex
@software{kimi_sae_2025,
  author = {Chen, Richard},
  title = {Sparse Autoencoder Training Pipeline for Kimi-K2-Thinking},
  year = {2025},
  url = {https://github.com/yourusername/kimi-sae-project}
}
```

## Support

- **Documentation**: README.md, USAGE.md
- **Issues**: GitHub Issues
- **Examples**: `examples/analyze_reasoning.py`
- **Contact**: [your email]

---

**Total Implementation**: 
- ~800 lines of Python code
- ~400 lines of bash scripts
- ~2000 lines of documentation
- Ready to run end-to-end
