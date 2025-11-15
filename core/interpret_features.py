"""
Feature Interpretation Script

Analyzes trained SAE features to understand what they represent.
Finds top-activating examples for specific features.

Usage:
    python interpret_features.py \
        --sae_path models/sae_layer45_8x/sae_best.pt \
        --data_dir data \
        --num_shards 4 \
        --feature_id 1234 \
        --top_k 10
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sae_model import load_sae
from utils.dataset import ActivationDataset, TokenIDDataset


def find_top_activating_examples(sae, dataset, token_dataset, feature_id, top_k=10, batch_size=8192):
    """
    Find examples where a specific feature has highest activation.
    
    Args:
        sae: Trained SAE model
        dataset: ActivationDataset
        token_dataset: TokenIDDataset
        feature_id: Which feature to analyze
        top_k: Number of top examples to return
        batch_size: Batch size for processing
    
    Returns:
        top_examples: List of (activation_value, token_idx, token_id) tuples
    """
    print(f"Finding top {top_k} activations for feature {feature_id}...")
    
    sae.eval()
    device = next(sae.parameters()).device
    
    # Store top activations using a min-heap approach
    top_activations = []
    
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # Load batch
        batch_acts = []
        for i in range(start_idx, end_idx):
            batch_acts.append(dataset[i])
        
        batch_acts = torch.stack(batch_acts).to(device)
        
        # Get feature activations
        with torch.no_grad():
            _, features = sae(batch_acts)
            feature_acts = features[:, feature_id].cpu().numpy()
        
        # Track top activations
        for i, act_val in enumerate(feature_acts):
            token_idx = start_idx + i
            
            if len(top_activations) < top_k:
                top_activations.append((act_val, token_idx))
                top_activations.sort(reverse=True)
            elif act_val > top_activations[-1][0]:
                top_activations[-1] = (act_val, token_idx)
                top_activations.sort(reverse=True)
        
        if batch_idx % 100 == 0:
            print(f"  Processed {batch_idx}/{n_batches} batches")
    
    # Get token IDs for top examples
    results = []
    for act_val, token_idx in top_activations:
        token_id = token_dataset[token_idx]
        results.append((act_val, token_idx, token_id))
    
    return results


def analyze_feature(sae, dataset, token_dataset, tokenizer, feature_id, top_k=10):
    """
    Analyze a specific feature by finding its top-activating examples.
    
    Args:
        sae: Trained SAE
        dataset: ActivationDataset
        token_dataset: TokenIDDataset  
        tokenizer: Tokenizer to decode token IDs
        feature_id: Feature to analyze
        top_k: Number of examples to show
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Feature {feature_id}")
    print(f"{'='*60}\n")
    
    # Find top activating examples
    top_examples = find_top_activating_examples(
        sae, dataset, token_dataset, feature_id, top_k
    )
    
    print(f"\nTop {top_k} Activating Examples:")
    print("-" * 60)
    
    for rank, (act_val, token_idx, token_id) in enumerate(top_examples, 1):
        # Decode token
        try:
            token_text = tokenizer.decode([token_id])
        except:
            token_text = f"<token_id={token_id}>"
        
        print(f"\nRank {rank}:")
        print(f"  Activation: {act_val:.4f}")
        print(f"  Token Index: {token_idx}")
        print(f"  Token ID: {token_id}")
        print(f"  Token Text: '{token_text}'")
    
    # Compute statistics
    activations = [act_val for act_val, _, _ in top_examples]
    print(f"\nStatistics:")
    print(f"  Max activation: {max(activations):.4f}")
    print(f"  Min activation (in top {top_k}): {min(activations):.4f}")
    print(f"  Mean activation (in top {top_k}): {np.mean(activations):.4f}")


def get_feature_statistics(sae, dataset, batch_size=8192, n_samples=100000):
    """
    Compute statistics about all features.
    
    Args:
        sae: Trained SAE
        dataset: ActivationDataset
        batch_size: Batch size
        n_samples: Number of samples to analyze
    
    Returns:
        dict with feature statistics
    """
    print(f"Computing feature statistics on {n_samples} samples...")
    
    sae.eval()
    device = next(sae.parameters()).device
    
    feature_max_acts = torch.zeros(sae.d_sae)
    feature_activations = torch.zeros(sae.d_sae)
    feature_counts = torch.zeros(sae.d_sae)
    
    n_batches = min((n_samples + batch_size - 1) // batch_size, 
                    (len(dataset) + batch_size - 1) // batch_size)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset), n_samples)
        
        # Load batch
        batch_acts = []
        for i in range(start_idx, end_idx):
            batch_acts.append(dataset[i])
        
        batch_acts = torch.stack(batch_acts).to(device)
        
        # Get features
        with torch.no_grad():
            _, features = sae(batch_acts)
            features = features.cpu()
        
        # Update statistics
        feature_max_acts = torch.maximum(feature_max_acts, features.max(dim=0)[0])
        feature_activations += features.sum(dim=0)
        feature_counts += (features > 0).sum(dim=0).float()
        
        if batch_idx % 10 == 0:
            print(f"  Processed {batch_idx}/{n_batches} batches")
    
    n_total = min(n_samples, len(dataset))
    
    stats = {
        'max_activations': feature_max_acts,
        'mean_activations': feature_activations / n_total,
        'activation_frequency': feature_counts / n_total,
        'n_samples': n_total,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Interpret SAE features")
    parser.add_argument("--sae_path", type=str, required=True,
                       help="Path to trained SAE checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Base data directory")
    parser.add_argument("--num_shards", type=int, default=4,
                       help="Number of shards")
    parser.add_argument("--feature_id", type=int, default=None,
                       help="Specific feature to analyze")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top examples to show")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--statistics", action="store_true",
                       help="Compute feature statistics")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAE Feature Interpretation")
    print("=" * 60)
    print(f"SAE: {args.sae_path}")
    print(f"Data: {args.data_dir}")
    print("")
    
    # Load SAE
    print("Loading SAE...")
    sae = load_sae(args.sae_path, device=args.device)
    print(f"  d_model: {sae.d_model}")
    print(f"  d_sae: {sae.d_sae}")
    print("")
    
    # Load datasets
    print("Loading datasets...")
    data_dirs = [f"{args.data_dir}/shard{i}" for i in range(args.num_shards)]
    
    act_dataset = ActivationDataset(data_dirs)
    token_dataset = TokenIDDataset(data_dirs)
    print("")
    
    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Thinking",
        trust_remote_code=True
    )
    print("")
    
    # Compute statistics if requested
    if args.statistics:
        stats = get_feature_statistics(sae, act_dataset)
        
        print("\n" + "=" * 60)
        print("Feature Statistics")
        print("=" * 60)
        
        # Find most active features
        top_features = torch.argsort(stats['max_activations'], descending=True)[:20]
        
        print("\nTop 20 Features by Max Activation:")
        print("-" * 60)
        for rank, feat_id in enumerate(top_features, 1):
            feat_id = feat_id.item()
            print(f"{rank:2d}. Feature {feat_id:5d}: "
                  f"max={stats['max_activations'][feat_id]:.4f}, "
                  f"mean={stats['mean_activations'][feat_id]:.6f}, "
                  f"freq={stats['activation_frequency'][feat_id]:.4f}")
        
        # Find most frequent features
        top_freq = torch.argsort(stats['activation_frequency'], descending=True)[:20]
        
        print("\nTop 20 Features by Activation Frequency:")
        print("-" * 60)
        for rank, feat_id in enumerate(top_freq, 1):
            feat_id = feat_id.item()
            print(f"{rank:2d}. Feature {feat_id:5d}: "
                  f"freq={stats['activation_frequency'][feat_id]:.4f}, "
                  f"max={stats['max_activations'][feat_id]:.4f}, "
                  f"mean={stats['mean_activations'][feat_id]:.6f}")
    
    # Analyze specific feature if requested
    if args.feature_id is not None:
        analyze_feature(
            sae, act_dataset, token_dataset, tokenizer,
            args.feature_id, args.top_k
        )
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
