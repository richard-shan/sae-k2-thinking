"""
Dataset utilities for loading activation data.

Implements memory-mapped dataset for efficient streaming of large
activation files without loading everything into RAM.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List


class ActivationDataset(Dataset):
    """
    Memory-mapped dataset for streaming activations from disk.
    
    This dataset uses memory mapping to avoid loading all 14+ TB of
    activations into RAM. Instead, it loads individual activation vectors
    on-demand during training.
    
    Args:
        data_dirs: List of shard directories (e.g., ['data/shard0', 'data/shard1'])
        transform: Optional transform to apply to activations
    """
    
    def __init__(self, data_dirs: List[str], transform=None):
        self.transform = transform
        self.chunks = []
        
        # Find all activation chunks across all shards
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            act_dir = data_path / "activations"
            
            if not act_dir.exists():
                print(f"Warning: {act_dir} does not exist, skipping")
                continue
            
            chunk_files = sorted(act_dir.glob("chunk_*.npy"))
            self.chunks.extend(chunk_files)
        
        if len(self.chunks) == 0:
            raise ValueError(f"No activation chunks found in {data_dirs}")
        
        print(f"Found {len(self.chunks)} activation chunks")
        
        # Memory-map all chunks (doesn't load into RAM)
        self.memmaps = []
        for chunk in self.chunks:
            mm = np.load(chunk, mmap_mode='r')
            self.memmaps.append(mm)
        
        # Calculate cumulative sizes for indexing
        self.chunk_sizes = [mm.shape[0] for mm in self.memmaps]
        self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        print(f"Total activations: {self.total_size:,}")
        print(f"Memory-mapped size: {self.total_size * 7168 * 2 / 1e12:.2f} TB")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Get a single activation vector.
        
        Args:
            idx: Index of activation to retrieve
        
        Returns:
            activation: Tensor of shape [7168]
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # Find which chunk this index belongs to
        chunk_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        # Load single activation vector from memory-mapped file
        activation = self.memmaps[chunk_idx][local_idx]
        activation = torch.from_numpy(activation.copy()).float()
        
        if self.transform is not None:
            activation = self.transform(activation)
        
        return activation
    
    def get_chunk_info(self):
        """Get information about chunks in dataset."""
        info = []
        for i, (chunk_path, size) in enumerate(zip(self.chunks, self.chunk_sizes)):
            info.append({
                'chunk_id': i,
                'path': str(chunk_path),
                'size': size,
                'start_idx': self.cumulative_sizes[i],
                'end_idx': self.cumulative_sizes[i+1]
            })
        return info


class TokenIDDataset(Dataset):
    """
    Memory-mapped dataset for token IDs corresponding to activations.
    
    This is useful for interpreting which tokens produced which activations
    during feature analysis.
    
    Args:
        data_dirs: List of shard directories
    """
    
    def __init__(self, data_dirs: List[str]):
        self.chunks = []
        
        # Find all token_id chunks across all shards
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            tok_dir = data_path / "token_ids"
            
            if not tok_dir.exists():
                print(f"Warning: {tok_dir} does not exist, skipping")
                continue
            
            chunk_files = sorted(tok_dir.glob("chunk_*.npy"))
            self.chunks.extend(chunk_files)
        
        if len(self.chunks) == 0:
            raise ValueError(f"No token_id chunks found in {data_dirs}")
        
        print(f"Found {len(self.chunks)} token_id chunks")
        
        # Memory-map all chunks
        self.memmaps = []
        for chunk in self.chunks:
            mm = np.load(chunk, mmap_mode='r')
            self.memmaps.append(mm)
        
        # Calculate cumulative sizes
        self.chunk_sizes = [mm.shape[0] for mm in self.memmaps]
        self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        print(f"Total token IDs: {self.total_size:,}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """Get token ID at index."""
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range")
        
        # Find chunk and local index
        chunk_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        # Load token ID
        token_id = self.memmaps[chunk_idx][local_idx]
        return int(token_id)
    
    def get_batch(self, indices):
        """
        Get multiple token IDs efficiently.
        
        Args:
            indices: List or tensor of indices
        
        Returns:
            token_ids: List of token IDs
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        token_ids = [self[idx] for idx in indices]
        return token_ids


def create_dataloaders(data_dirs, batch_size=8192, train_split=0.98, num_workers=4):
    """
    Create train and validation dataloaders from activation data.
    
    Args:
        data_dirs: List of shard directories
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    dataset = ActivationDataset(data_dirs)
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset):,}")
    print(f"Val size: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader
