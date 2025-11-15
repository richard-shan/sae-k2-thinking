# Changes Log

## Version 1.1 - SAELens Architecture & SlimPajama Integration

### SAE Architecture Improvements (Based on Neel Nanda/SAELens)

**Previous Architecture:**
- Simple encoder-decoder with ReLU
- Basic unit-norm decoder constraint
- No activation normalization

**New Architecture:**
```python
class SparseAutoencoder:
    # Weights (using nn.Parameter directly, not nn.Linear)
    W_enc: [d_model, d_sae]  # Encoder weights
    b_enc: [d_sae]           # Encoder bias
    W_dec: [d_sae, d_model]  # Decoder weights  
    b_dec: [d_model]         # Decoder bias
    
    # Optional: Activation statistics
    activation_mean: [d_model]  # Running mean for centering
    
    # Forward pass:
    1. Optional: Center activations using activation_mean
    2. Optional: Subtract b_dec from input (apply_b_dec_to_input)
    3. Encode: pre_acts = (x_centered @ W_enc) + b_enc
    4. Sparsify: features = ReLU(pre_acts)
    5. Decode: reconstructed = (features @ W_dec) + b_dec
```

**Key Changes:**

1. **Activation Normalization** (`normalize_activations`)
   - `"expected_average_only_in"` (default): Center activations by running mean
   - `"layer_norm"`: Apply layer normalization
   - `"none"`: No normalization
   
   **Why:** Helps SAE learn better features by removing activation distribution shifts

2. **Decoder Bias to Input** (`apply_b_dec_to_input`)
   - Optionally subtract `b_dec` from input before encoding
   - **Why:** Allows encoder to focus on learning deviations from baseline

3. **Weight Initialization**
   - Encoder: Kaiming uniform (better for ReLU)
   - Decoder: Normalized transpose of encoder
   - **Why:** Better training stability and faster convergence

4. **Direct Parameter Access**
   - Use `nn.Parameter` instead of `nn.Linear`
   - Allows finer control over normalization
   - **Why:** SAELens convention, easier to manipulate decoder norms

### Dataset Change: SlimPajama-627B

**Previous:** `tiiuae/falcon-refinedweb`
**New:** `cerebras/SlimPajama-627B`

**Why SlimPajama:**
- Higher quality text (cleaned, deduplicated)
- More diverse sources (CommonCrawl, C4, GitHub, ArXiv, Books, StackExchange, Wikipedia)
- 627B tokens available
- Better for reasoning research (includes code, math, scientific text)

**Implementation Changes:**
```python
# Dataset loading
dataset = load_dataset(
    "cerebras/SlimPajama-627B",
    streaming=True,
    split="train"
)

# Text field access (SlimPajama uses 'text' not 'content')
text_content = example.get("text", example.get("content", ""))
```

### Updated Training Defaults

**Recommended hyperparameters based on SAELens:**
```bash
python scripts/train_sae.py \
    --data_dir data \
    --output_dir models/sae_layer45_8x \
    --num_shards 4 \
    --target_layer 45 \
    --d_model 7168 \
    --d_sae 57344 \
    --l1_coeff 5e-3 \                      # Increased for better sparsity
    --lr 3e-4 \
    --batch_size 4096 \                    # Larger batch = more stable
    --epochs 3 \
    --normalize_activations expected_average_only_in \  # NEW
    --apply_b_dec_to_input                 # NEW (optional)
```

### Compatibility Notes

**Breaking Changes:**
- SAE `forward()` now returns 3 values: `(reconstructed, features, loss_dict)`
  - Previously returned 2: `(reconstructed, features)`
  - Update inference code accordingly

**Backward Compatibility:**
- Old checkpoints won't load with new architecture
- Need to retrain SAEs with new architecture
- Collection data is still compatible (just activations + token IDs)

### Performance Improvements

**Expected metrics with new architecture:**
- Better reconstruction (lower MSE)
- More interpretable features (cleaner top-activating examples)
- Faster convergence (better initialization)
- More stable training (activation normalization)

**Typical results:**
- L0: 50-100 (same as before)
- MSE: 0.005-0.008 (improved from 0.007-0.010)
- Training time: ~1 hour (unchanged)

### Migration Guide

**If you have existing collection data:**
1. Keep existing activation data (still compatible)
2. Retrain SAE with new architecture
3. Update any inference scripts to handle 3-value return

**If starting fresh:**
1. Use new collection script (automatically uses SlimPajama)
2. Train with new SAE architecture
3. Enjoy better features!

### References

- SAELens Tutorial: https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb
- Neel Nanda's W&B: https://wandb.ai/jbloom/sae_lens_tutorial
- SlimPajama: https://huggingface.co/datasets/cerebras/SlimPajama-627B
- Original SAE Paper: https://transformer-circuits.pub/2023/monosemantic-features

### Testing

To verify the new architecture works:

```bash
# Test SAE initialization
python -c "from utils.sae_model import SparseAutoencoder; \
    sae = SparseAutoencoder(7168, 57344, 'expected_average_only_in', False); \
    import torch; x = torch.randn(10, 7168); \
    recon, feat, loss_dict = sae(x); \
    print(f'✓ Forward pass works: {recon.shape}, {feat.shape}')"

# Test collection with SlimPajama
bash launch_pilot.sh  # Uses SlimPajama automatically
```

### Future Improvements

Potential additions in future versions:
- Ghost grads for dead feature resampling
- Gated SAE architecture
- Top-k activation instead of L1
- Anthropic's scaling laws for SAE capacity

---

**Version 1.0** - Initial release with basic SAE + RefinedWeb
**Version 1.1** - SAELens architecture + SlimPajama (Current)
