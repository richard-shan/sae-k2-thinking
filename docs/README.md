# Kimi-K2 Sparse Autoencoder Project

Train Sparse Autoencoders (SAEs) on Kimi-K2-Thinking to discover interpretable reasoning circuits.

## Overview

This project implements a two-phase pipeline:
1. **Phase 1: Activation Collection** - Extract internal activations from Kimi-K2-Thinking across 1B tokens
2. **Phase 2: SAE Training** - Train sparse autoencoders to learn interpretable feature representations

The resulting SAE can be used to study reasoning mechanisms in the K2 model across tasks like GSM8K, MATH, GPQA, and chain-of-thought reasoning.

## Requirements

### Hardware
- **Phase 1 (Collection)**: 8× H100 (80GB) or 4× A100 (80GB) - ~4 hours
- **Phase 2 (Training)**: 1× H100 or A100 - ~1 hour
- **Storage**: 15+ TB SSD for activation data

### Software
- Python 3.9+
- CUDA 11.8+
- See `requirements.txt` for Python packages

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/kimi-sae-project.git
cd kimi-sae-project

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{shard0,shard1,shard2,shard3}/{activations,token_ids}
mkdir -p logs models
```

### 2. Run Pilot Collection (5-10 minutes)

Test the pipeline on 10M tokens before committing to full collection:

```bash
bash launch_pilot.sh
```

**Expected output:**
```
Shard 0: Loading model...
Shard 0: Model loaded successfully
Shard 0: Starting collection...
Shard 0: 10,000,000 tokens collected
Shard 0: Collection complete!
✓ All checks passed!
```

### 3. Run Full Collection (3-4 hours)

If pilot succeeds, launch full 1B token collection:

```bash
bash launch_collection.sh
```

**Monitor progress in another terminal:**
```bash
bash monitor.sh
```

### 4. Train SAE (1 hour)

After collection completes:

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

### 5. Download Results

```bash
# Compress trained SAE
cd models/sae_layer45_8x
tar -czf sae_trained.tar.gz sae_best.pt metadata.json

# Download to local machine
scp -i ~/.ssh/key.pem ubuntu@<ip>:~/kimi-sae-project/models/sae_layer45_8x/sae_trained.tar.gz ./
```

## Repository Structure

```
kimi-sae-project/
├── scripts/
│   ├── collect_activations.py    # Main collection script
│   ├── train_sae.py              # SAE training script
│   ├── verify_collection.py      # Data validation
│   └── interpret_features.py     # Feature interpretation
├── utils/
│   ├── __init__.py
│   ├── sae_model.py             # SAE architecture
│   └── dataset.py               # Activation dataset loader
├── launch_pilot.sh               # Quick 10M token test
├── launch_collection.sh          # Full 1B token collection
├── monitor.sh                    # Progress monitoring
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

### Adjusting Collection Parameters

Edit `launch_collection.sh`:
- `--target_layer`: Which layer to extract (default: 45)
- `--tokens_per_shard`: Tokens per parallel stream (default: 250M)
- `--chunk_size`: Save frequency (default: 10M tokens)
- `--max_length`: Max sequence length (default: 2048)

### Adjusting SAE Parameters

Edit `train_sae.py` arguments:
- `--d_sae`: SAE hidden dimension (default: 57344 = 8× expansion)
- `--l1_coeff`: Sparsity penalty (default: 1e-3)
- `--lr`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 8192)
- `--epochs`: Training epochs (default: 3)

## Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Pre-download model
huggingface-cli download moonshotai/Kimi-K2-Thinking
```

### Out of Memory

If CUDA OOM errors occur during collection:
```python
# In collect_activations.py, reduce batch_size:
--batch_size 1  # Process one example at a time
```

### Checkpoint Recovery

If collection crashes, it will automatically resume from the last checkpoint:
```bash
# Check progress
cat data/shard0/checkpoint.json

# Resume by re-running
bash launch_collection.sh
```

### Verification Failures

If verification fails:
```bash
# Check logs for specific errors
tail -100 logs/shard0.log

# Verify individual shard
python scripts/verify_collection.py --data_dir data/shard0
```

## Cost Estimates

### Lambda Labs 8× H100 ($23.92/hour)

| Phase | Time | Cost |
|-------|------|------|
| Pilot (10M tokens) | 5 min | $2 |
| Collection (1B tokens) | 4 hours | $96 |
| SAE Training | 1 hour | $24 |
| **Total** | **~5 hours** | **~$120** |

### Alternative: Separate Training

1. Run collection on 8× H100: $96 (4 hours)
2. Shut down cluster
3. Rent 1× H100 for training: $5 (1 hour)
4. **Total: $101**

## Dataset Details

### Activation Format
- **File type**: NumPy `.npy` arrays
- **Precision**: FP16 (float16)
- **Shape**: `[n_tokens, 7168]`
- **Size**: ~14.3 GB per 1M tokens

### Token ID Format
- **File type**: NumPy `.npy` arrays
- **Precision**: int32
- **Shape**: `[n_tokens]`
- **Size**: ~4 MB per 1M tokens

### Storage Layout
```
data/
├── shard0/
│   ├── activations/
│   │   ├── chunk_0000.npy  # 10M tokens × 7168 dim × 2 bytes = 143 GB
│   │   ├── chunk_0001.npy
│   │   └── ...
│   ├── token_ids/
│   │   ├── chunk_0000.npy  # 10M tokens × 4 bytes = 40 MB
│   │   └── ...
│   ├── metadata.json
│   └── checkpoint.json
└── shard1/ ... shard2/ ... shard3/
```

## Using the Trained SAE

```python
from utils.sae_model import load_sae
import torch

# Load trained SAE
sae = load_sae("models/sae_layer45_8x/sae_best.pt", device="cuda")

# Get activation from K2
activation = torch.randn(1, 7168).cuda()  # Example activation

# Encode with SAE
with torch.no_grad():
    reconstructed, features = sae(activation)

# Find active features
active_features = (features > 0).nonzero(as_tuple=True)[1]
print(f"Active features: {active_features.tolist()}")
print(f"Sparsity (L0): {len(active_features)}")
```

## Research Applications

With the trained SAE, you can:

1. **Identify reasoning features**: Which features activate during mathematical reasoning vs. logical reasoning?
2. **Study reasoning circuits**: How do features compose during chain-of-thought generation?
3. **Analyze complexity scaling**: How do feature activations change from simple to complex problems?
4. **Perform causal interventions**: Ablate specific features and measure impact on reasoning quality
5. **Compare reasoning modalities**: Do calculation-heavy and explanation-heavy approaches use different features?

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{kimi_sae_2025,
  author = {Your Name},
  title = {Sparse Autoencoder Training Pipeline for Kimi-K2-Thinking},
  year = {2025},
  url = {https://github.com/yourusername/kimi-sae-project}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Moonshot AI for releasing Kimi-K2-Thinking
- Anthropic for SAE research and methodologies
- HuggingFace for model hosting and transformers library

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
