"""
SAE Training Script

Trains a sparse autoencoder on collected activations from Kimi-K2-Thinking.

Usage:
    python train_sae.py \
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
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sae_model import SparseAutoencoder, save_sae
from utils.dataset import create_dataloaders


def train_epoch(sae, train_loader, optimizer, l1_coeff, device):
    """Train for one epoch."""
    sae.train()
    
    total_recon_loss = 0
    total_l1_loss = 0
    total_l0 = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        reconstructed, feature_acts, loss_dict = sae(batch)
        
        # Compute losses
        recon_loss = F.mse_loss(reconstructed, batch)
        l1_loss = feature_acts.abs().mean()
        loss = recon_loss + l1_coeff * l1_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize decoder columns
        sae.normalize_decoder()
        
        # Track metrics
        total_recon_loss += recon_loss.item()
        total_l1_loss += l1_loss.item()
        total_l0 += (feature_acts > 0).float().sum(dim=1).mean().item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'recon': f'{recon_loss.item():.6f}',
            'l0': f'{(feature_acts > 0).float().sum(dim=1).mean().item():.1f}'
        })
    
    return {
        'recon_loss': total_recon_loss / n_batches,
        'l1_loss': total_l1_loss / n_batches,
        'l0': total_l0 / n_batches
    }


def validate(sae, val_loader, l1_coeff, device):
    """Validate the model."""
    sae.eval()
    
    total_recon_loss = 0
    total_l1_loss = 0
    total_l0 = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, feature_acts, loss_dict = sae(batch)
            
            # Compute losses
            recon_loss = F.mse_loss(reconstructed, batch)
            l1_loss = feature_acts.abs().mean()
            
            # Track metrics
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
            total_l0 += (feature_acts > 0).float().sum(dim=1).mean().item()
            n_batches += 1
    
    return {
        'recon_loss': total_recon_loss / n_batches,
        'l1_loss': total_l1_loss / n_batches,
        'l0': total_l0 / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Base directory containing shard0, shard1, etc.")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save trained model")
    parser.add_argument("--num_shards", type=int, default=4,
                       help="Number of shards to load")
    
    # Model arguments
    parser.add_argument("--target_layer", type=int, default=45,
                       help="Which layer the activations are from")
    parser.add_argument("--d_model", type=int, default=7168,
                       help="Activation dimension")
    parser.add_argument("--d_sae", type=int, default=57344,
                       help="SAE hidden dimension (8x expansion)")
    parser.add_argument("--normalize_activations", type=str, 
                       default="expected_average_only_in",
                       choices=["none", "expected_average_only_in", "layer_norm"],
                       help="Activation normalization strategy")
    parser.add_argument("--apply_b_dec_to_input", action="store_true",
                       help="Apply decoder bias to input")
    
    # Training arguments
    parser.add_argument("--l1_coeff", type=float, default=1e-3,
                       help="L1 sparsity coefficient")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8192,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--train_split", type=float, default=0.98,
                       help="Fraction of data for training")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Kimi-K2 SAE Training")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Target layer: {args.target_layer}")
    print(f"Model dimensions: {args.d_model} -> {args.d_sae} (expansion: {args.d_sae/args.d_model:.1f}x)")
    print(f"L1 coefficient: {args.l1_coeff}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("")
    
    # Load dataset
    print("Loading activation dataset...")
    data_dirs = [f"{args.data_dir}/shard{i}" for i in range(args.num_shards)]
    
    try:
        train_loader, val_loader = create_dataloaders(
            data_dirs=data_dirs,
            batch_size=args.batch_size,
            train_split=args.train_split,
            num_workers=args.num_workers
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("")
    
    # Initialize SAE
    print(f"Initializing SAE...")
    sae = SparseAutoencoder(
        d_model=args.d_model,
        d_sae=args.d_sae,
        normalize_activations=args.normalize_activations,
        apply_b_dec_to_input=args.apply_b_dec_to_input,
    ).to(args.device)
    
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"  Encoder: {args.d_model} -> {args.d_sae}")
    print(f"  Decoder: {args.d_sae} -> {args.d_model}")
    print("")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    print("Starting training...")
    print("")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            sae, train_loader, optimizer, args.l1_coeff, args.device
        )
        
        # Validate
        val_metrics = validate(sae, val_loader, args.l1_coeff, args.device)
        
        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train - Recon Loss: {train_metrics['recon_loss']:.6f}, "
              f"L1: {train_metrics['l1_loss']:.6f}, L0: {train_metrics['l0']:.1f}")
        print(f"  Val   - Recon Loss: {val_metrics['recon_loss']:.6f}, "
              f"L1: {val_metrics['l1_loss']:.6f}, L0: {val_metrics['l0']:.1f}")
        
        # Save metrics
        training_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Save best model
        if val_metrics['recon_loss'] < best_val_loss:
            best_val_loss = val_metrics['recon_loss']
            
            print(f"  ✓ New best model! (val_loss={best_val_loss:.6f})")
            
            # Save model
            save_sae(
                sae,
                output_path / "sae_best.pt",
                metadata={
                    'epoch': epoch + 1,
                    'val_loss': val_metrics['recon_loss'],
                    'val_l0': val_metrics['l0'],
                    'train_loss': train_metrics['recon_loss'],
                    'train_l0': train_metrics['l0'],
                    'l1_coeff': args.l1_coeff,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                }
            )
        
        print("")
    
    # Save final metadata
    metadata = {
        "model_config": {
            "d_model": args.d_model,
            "d_sae": args.d_sae,
            "expansion_factor": args.d_sae / args.d_model,
        },
        "training_config": {
            "l1_coeff": args.l1_coeff,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "train_split": args.train_split,
        },
        "data_info": {
            "num_shards": args.num_shards,
            "target_layer": args.target_layer,
            "source_model": "moonshotai/Kimi-K2-Thinking",
        },
        "best_results": {
            "val_loss": best_val_loss,
            "val_l0": val_metrics['l0'],
        },
        "training_history": training_history,
        "trained_on": str(datetime.now()),
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation L0: {val_metrics['l0']:.1f}")
    print(f"Model saved to: {output_path / 'sae_best.pt'}")
    print(f"Metadata saved to: {output_path / 'metadata.json'}")


if __name__ == "__main__":
    main()
