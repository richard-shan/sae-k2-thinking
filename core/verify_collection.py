"""
Data Verification Script

Validates collected activation data for integrity and completeness.

Usage:
    python verify_collection.py --data_dir data/shard0
    python verify_collection.py --data_dir data/shard0 --data_dir data/shard1
"""

import argparse
from pathlib import Path
import numpy as np
import json


def verify_shard(data_dir, verbose=True):
    """
    Verify data integrity for a single shard.
    
    Args:
        data_dir: Path to shard directory
        verbose: Whether to print detailed information
    
    Returns:
        bool: True if verification passed, False otherwise
    """
    data_path = Path(data_dir)
    
    if verbose:
        print(f"\nVerifying {data_dir}...")
        print("-" * 60)
    
    # Check metadata
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        if verbose:
            print("  ✗ metadata.json not found")
        return False
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    if verbose:
        print(f"  ✓ Metadata found")
        print(f"    Target layer: {metadata.get('target_layer', 'N/A')}")
        print(f"    Target tokens: {metadata.get('tokens_per_shard', 'N/A'):,}")
        if 'tokens_collected' in metadata:
            print(f"    Collected tokens: {metadata['tokens_collected']:,}")
    
    # Find activation and token_id chunks
    act_dir = data_path / "activations"
    tok_dir = data_path / "token_ids"
    
    if not act_dir.exists():
        if verbose:
            print(f"  ✗ Activations directory not found: {act_dir}")
        return False
    
    if not tok_dir.exists():
        if verbose:
            print(f"  ✗ Token IDs directory not found: {tok_dir}")
        return False
    
    act_chunks = sorted(act_dir.glob("chunk_*.npy"))
    tok_chunks = sorted(tok_dir.glob("chunk_*.npy"))
    
    if verbose:
        print(f"  ✓ Found {len(act_chunks)} activation chunks")
        print(f"  ✓ Found {len(tok_chunks)} token_id chunks")
    
    if len(act_chunks) == 0:
        if verbose:
            print("  ✗ No activation chunks found!")
        return False
    
    if len(act_chunks) != len(tok_chunks):
        if verbose:
            print("  ✗ Mismatch in number of chunks!")
            print(f"    Activations: {len(act_chunks)}")
            print(f"    Token IDs: {len(tok_chunks)}")
        return False
    
    # Verify each chunk
    total_tokens = 0
    chunk_errors = []
    
    for i, (act_file, tok_file) in enumerate(zip(act_chunks, tok_chunks)):
        try:
            # Load with memory mapping (doesn't load full file)
            acts = np.load(act_file, mmap_mode='r')
            toks = np.load(tok_file, mmap_mode='r')
            
            # Check shapes
            if acts.ndim != 2:
                chunk_errors.append(f"Chunk {i}: Invalid activation dimensions (expected 2D, got {acts.ndim}D)")
                continue
            
            if acts.shape[1] != 7168:
                chunk_errors.append(f"Chunk {i}: Wrong activation dimension {acts.shape[1]} (expected 7168)")
                continue
            
            if acts.shape[0] != toks.shape[0]:
                chunk_errors.append(f"Chunk {i}: Shape mismatch (acts: {acts.shape[0]}, toks: {toks.shape[0]})")
                continue
            
            # Check data types
            if acts.dtype != np.float16:
                chunk_errors.append(f"Chunk {i}: Wrong activation dtype {acts.dtype} (expected float16)")
            
            if toks.dtype != np.int32:
                chunk_errors.append(f"Chunk {i}: Wrong token_id dtype {toks.dtype} (expected int32)")
            
            total_tokens += acts.shape[0]
            
        except Exception as e:
            chunk_errors.append(f"Chunk {i}: Error loading - {str(e)}")
    
    if chunk_errors:
        if verbose:
            print("  ✗ Chunk verification errors:")
            for error in chunk_errors:
                print(f"    - {error}")
        return False
    
    if verbose:
        print(f"  ✓ All chunks verified successfully")
        print(f"  ✓ Total tokens: {total_tokens:,}")
        print(f"  ✓ Activation shape: (n_tokens, 7168)")
        print(f"  ✓ Data types correct (float16, int32)")
        
        # Storage size
        act_size_gb = total_tokens * 7168 * 2 / 1e9
        tok_size_mb = total_tokens * 4 / 1e6
        print(f"  ✓ Storage used:")
        print(f"    Activations: {act_size_gb:.2f} GB")
        print(f"    Token IDs: {tok_size_mb:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify collected activation data")
    parser.add_argument("--data_dir", action="append", required=True,
                       help="Shard directory to verify (can specify multiple)")
    parser.add_argument("--quiet", action="store_true",
                       help="Only print summary, not details")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Activation Data Verification")
    print("=" * 60)
    
    results = {}
    for data_dir in args.data_dir:
        results[data_dir] = verify_shard(data_dir, verbose=not args.quiet)
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for data_dir, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{data_dir}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✓ ALL SHARDS VERIFIED SUCCESSFULLY")
        print("\nNext step: Train SAE")
        print("  python scripts/train_sae.py --data_dir data --output_dir models/sae")
    else:
        print("✗ SOME SHARDS FAILED VERIFICATION")
        print("\nPlease check error messages above and re-run collection if needed.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
