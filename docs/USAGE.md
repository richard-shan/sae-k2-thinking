# Detailed Usage Guide

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Phase 1: Activation Collection](#phase-1-activation-collection)
3. [Phase 2: SAE Training](#phase-2-sae-training)
4. [Phase 3: Feature Interpretation](#phase-3-feature-interpretation)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Initial Setup

### Prerequisites
- 8× H100 (80GB) or 4× A100 (80GB) GPUs
- 15+ TB SSD storage
- Ubuntu 20.04+ with CUDA 11.8+
- Python 3.9+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/kimi-sae-project.git
cd kimi-sae-project

# Run setup script
bash setup.sh
```

The setup script will:
- Check Python and CUDA versions
- Install dependencies
- Create necessary directories
- Make scripts executable
- Test imports

---

## Phase 1: Activation Collection

### Step 1: Pilot Collection (5-10 minutes)

**Purpose**: Test the pipeline on 10M tokens before committing to full collection.

```bash
bash launch_pilot.sh
```

**What it does**:
- Downloads Kimi-K2-Thinking model (~10 min first time)
- Collects 10M tokens of activations
- Saves 2 chunks (~1.4 GB) to `data/pilot/`
- Automatically verifies data integrity

**Expected output**:
```
Shard 0: Loading model...
Shard 0: Model loaded successfully
Shard 0: 5,000,000 tokens collected
Shard 0: Saved chunk 0
Shard 0: 10,000,000 tokens collected
Shard 0: Collection complete!
✓ All checks passed!
```

**If pilot fails**: Check `logs/pilot.log` for errors. Common issues:
- CUDA OOM: Model too large for GPUs
- Disk space: Not enough storage
- Network: Model download timeout

### Step 2: Full Collection (3-4 hours)

**Purpose**: Collect 1B tokens across 4 parallel shards.

```bash
bash launch_collection.sh
```

**What it does**:
- Launches 4 parallel processes (one per GPU pair)
- Each process collects 250M tokens
- Saves checkpoints every 10M tokens
- Total: 1B tokens, ~14.3 TB

**Monitoring in real-time**:

Open a second terminal and run:
```bash
bash monitor.sh
```

**Monitor output**:
```
Shard 0: 87,432,100 / 250,000,000 tokens (35.0%) - Chunk 8
Shard 1: 92,145,890 / 250,000,000 tokens (36.9%) - Chunk 9
Shard 2: 85,234,120 / 250,000,000 tokens (34.1%) - Chunk 8
Shard 3: 89,012,340 / 250,000,000 tokens (35.6%) - Chunk 8

Disk usage:
1.2T    data/shard0
1.3T    data/shard1
...
```

**Checkpoint recovery**:

If a shard crashes, simply re-run `launch_collection.sh`. Each shard will automatically resume from its last checkpoint.

### Step 3: Verification

After collection completes:

```bash
python scripts/verify_collection.py \
    --data_dir data/shard0 \
    --data_dir data/shard1 \
    --data_dir data/shard2 \
    --data_dir data/shard3
```

**Expected output**:
```
✓ Found 25 activation chunks per shard
✓ Total tokens: 1,000,000,000
✓ Activation shape verified: (n_tokens, 7168)
✓ All chunks valid!
```

---

## Phase 2: SAE Training

### Training the SAE (1 hour)

After collecting activations, train the sparse autoencoder:

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

**Parameters explained**:
- `--data_dir`: Base directory containing shard0, shard1, etc.
- `--output_dir`: Where to save trained model
- `--num_shards`: Number of shards (4 for standard setup)
- `--target_layer`: Which K2 layer activations are from (45)
- `--d_model`: Activation dimension (7168 for K2)
- `--d_sae`: SAE hidden dimension (57344 = 8× expansion)
- `--l1_coeff`: Sparsity penalty (higher = sparser features)
- `--lr`: Learning rate
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs

**Training output**:
```
Loading activation dataset...
Found 100 activation chunks
Total activations: 1,000,000,000
Train size: 980,000,000
Val size: 20,000,000

Epoch 1/3
Training: 100%|██████████| 119628/119628 [45:23<00:00]
Validating: 100%|██████████| 2441/2441 [01:12<00:00]

Epoch 1:
  Train - Loss: 0.007123, L0: 65.4
  Val   - Loss: 0.007089, L0: 64.8
  ✓ New best model! (val_loss=0.007089)
...
```

**Output files**:
- `models/sae_layer45_8x/sae_best.pt` - Trained SAE weights (~3 GB)
- `models/sae_layer45_8x/metadata.json` - Training configuration and metrics

### Tuning Hyperparameters

**Sparsity (L1 coefficient)**:

Target L0 (number of active features per token): 50-100

- L0 too high (>150): Increase `--l1_coeff` (try 3e-3)
- L0 too low (<30): Decrease `--l1_coeff` (try 3e-4)

**Reconstruction quality**:

Target MSE loss: 0.005-0.010

- Loss too high (>0.015): Train longer or decrease `--l1_coeff`
- Loss too low (<0.003): May be overfitting, increase `--l1_coeff`

**Expansion factor**:

- 4× (28,672 features): Faster training, less sparse
- 8× (57,344 features): Standard, good balance
- 16× (114,688 features): More features, better separation

---

## Phase 3: Feature Interpretation

### Analyzing Trained Features

#### Get overall statistics:

```bash
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --statistics
```

**Output**:
```
Top 20 Features by Max Activation:
 1. Feature 12845: max=45.3421, mean=0.0234, freq=0.1234
 2. Feature  3921: max=42.1893, mean=0.0189, freq=0.0923
...

Top 20 Features by Activation Frequency:
 1. Feature  1023: freq=0.8234, max=12.3421, mean=0.5234
...
```

#### Analyze specific feature:

```bash
python scripts/interpret_features.py \
    --sae_path models/sae_layer45_8x/sae_best.pt \
    --data_dir data \
    --num_shards 4 \
    --feature_id 12845 \
    --top_k 20
```

**Output**:
```
Analyzing Feature 12845
========================================

Top 20 Activating Examples:

Rank 1:
  Activation: 45.3421
  Token Index: 123456789
  Token ID: 1234
  Token Text: ' derivative'

Rank 2:
  Activation: 44.8923
  Token Index: 234567890
  Token ID: 5678
  Token Text: ' calculus'
...
```

### Interpreting Features for Reasoning

To study reasoning circuits, run the SAE on reasoning datasets:

```python
from utils.sae_model import load_sae
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load K2 and SAE
model = AutoModelForCausalLM.from_pretrained("moonshotai/Kimi-K2-Thinking", ...)
sae = load_sae("models/sae_layer45_8x/sae_best.pt")

# Hook to capture activations
activations = []
def hook(module, input, output):
    activations.append(output[0])

hook_handle = model.model.layers[45].register_forward_hook(hook)

# Run on reasoning problem
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", ...)
text = "Solve: 2x + 5 = 13"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    _ = model(**inputs)
    acts = activations[0]
    _, features = sae(acts)

# Analyze which features fire
active_features = (features > 0).nonzero(as_tuple=True)[1]
print(f"Active features: {active_features.tolist()}")
```

---

## Troubleshooting

### Collection Issues

**CUDA Out of Memory**:
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size in `collect_activations.py`:
```bash
python scripts/collect_activations.py ... --batch_size 1
```

**Model download timeout**:
```
ConnectionError: Failed to download model
```
Solution: Pre-download model:
```bash
huggingface-cli download moonshotai/Kimi-K2-Thinking
```

**Disk full during collection**:
```
OSError: [Errno 28] No space left on device
```
Solution: 
- Delete pilot data: `rm -rf data/pilot`
- Check with: `df -h`
- Collection saves checkpoints every 10M tokens, so you can resume after freeing space

### Training Issues

**DataLoader workers crash**:
```
RuntimeError: DataLoader worker crashed
```
Solution: Reduce number of workers:
```bash
python scripts/train_sae.py ... --num_workers 0
```

**Training too slow**:
- Increase batch size: `--batch_size 16384`
- Use fewer workers: `--num_workers 2`
- Use only validation subset during development

**Poor reconstruction**:
- Decrease L1 coefficient: `--l1_coeff 5e-4`
- Train longer: `--epochs 5`
- Check data quality with verification script

---

## Advanced Usage

### Multiple Layer Collection

Collect activations from multiple layers:

```bash
# Layer 30 (early reasoning)
bash launch_collection.sh  # modify target_layer to 30

# Layer 45 (mid reasoning) 
bash launch_collection.sh  # target_layer 45

# Layer 55 (late reasoning)
bash launch_collection.sh  # target_layer 55
```

### Training Multiple SAEs

Train SAEs with different expansion factors:

```bash
# 4× expansion
python scripts/train_sae.py ... --d_sae 28672 --output_dir models/sae_4x

# 8× expansion (standard)
python scripts/train_sae.py ... --d_sae 57344 --output_dir models/sae_8x

# 16× expansion
python scripts/train_sae.py ... --d_sae 114688 --output_dir models/sae_16x
```

### Using Different Corpora

Modify `collect_activations.py` to use different datasets:

```python
# Wikipedia
dataset = load_dataset("wikipedia", "20220301.en", streaming=True)

# Books
dataset = load_dataset("bookcorpus", streaming=True)

# Code
dataset = load_dataset("codeparrot/github-code", streaming=True)

# Math-heavy text
dataset = load_dataset("EleutherAI/proof-pile-2", streaming=True)
```

### Distributed Training

For even faster SAE training, use PyTorch DDP:

```bash
torchrun --nproc_per_node=4 scripts/train_sae.py ...
```

---

## Cost Optimization

### Minimize costs by:

1. **Run pilot first**: Catch issues before full collection ($2 vs $100)

2. **Separate phases**: 
   - Collection: 8× H100 ($96)
   - Training: 1× H100 ($5)
   - Total: $101 instead of $120

3. **Use spot instances**: Can save 50-70% on cloud providers

4. **Batch multiple experiments**: Collect once, train multiple SAEs

---

## Research Workflow

Typical research pipeline:

1. **Pilot** → Validate setup (5 min, $2)
2. **Collection** → Gather 1B activations (4 hrs, $96)
3. **Baseline SAE** → Train 8× expansion (1 hr, $5)
4. **Interpretation** → Identify interesting features (local)
5. **Ablation studies** → Test different hyperparameters (local)
6. **Reasoning analysis** → Run on GSM8K, MATH, etc. (local)
7. **Paper writing** → Document findings (local)

Total cloud cost: ~$103

---

For questions or issues, open a GitHub issue or contact the maintainer.
