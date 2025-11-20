"""
Activation Collection Script for Kimi-K2-Thinking

Extracts internal layer activations during forward passes over a large text corpus.
Supports distributed collection across multiple GPU pairs for parallel processing.

Usage:
    python collect_activations.py \
        --shard_id 0 \
        --num_shards 4 \
        --output_dir data/shard0 \
        --target_layer 45 \
        --tokens_per_shard 250000000 \
        --chunk_size 10000000
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from pathlib import Path
import json
from datetime import datetime
import sys

try:
    from PIL import Image as _PILImage

    if not hasattr(_PILImage, "Resampling"):
        class _ResamplingProxy:
            NEAREST = _PILImage.NEAREST
            BOX = getattr(_PILImage, "BOX", _PILImage.NEAREST)
            BILINEAR = _PILImage.BILINEAR
            HAMMING = getattr(_PILImage, "HAMMING", _PILImage.BILINEAR)
            BICUBIC = _PILImage.BICUBIC
            LANCZOS = getattr(
                _PILImage,
                "LANCZOS",
                getattr(_PILImage, "ANTIALIAS", _PILImage.BICUBIC),
            )

        _PILImage.Resampling = _ResamplingProxy  # type: ignore[attr-defined]
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Collect activations from Kimi-K2-Thinking")
    parser.add_argument("--shard_id", type=int, required=True,
                       help="Shard ID (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=4,
                       help="Total number of parallel shards")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save activations")
    parser.add_argument("--target_layer", type=int, default=45,
                       help="Which layer to extract activations from")
    parser.add_argument("--tokens_per_shard", type=int, default=250_000_000,
                       help="Number of tokens to collect per shard")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                       help="Number of tokens per saved chunk (controls memory usage)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-K2-Thinking",
                       help="HuggingFace model name")
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split to use")
    args = parser.parse_args()
    
    # Guard against unrealistic chunk sizes that would exhaust RAM
    MAX_CHUNK_TOKENS = 1_000_000  # ~14 GB at 7168 dim FP16
    if args.chunk_size > MAX_CHUNK_TOKENS:
        est_gb = args.chunk_size * 7168 * 2 / 1e9
        print(f"ERROR: chunk_size={args.chunk_size:,} tokens would require ~{est_gb:.1f} GB RAM per shard.")
        print(f"Please lower --chunk_size (recommended <= {MAX_CHUNK_TOKENS:,}).")
        sys.exit(1)
    
    # Setup directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "activations").mkdir(exist_ok=True)
    (output_path / "token_ids").mkdir(exist_ok=True)
    
    print(f"Shard {args.shard_id}: Initializing...")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Tokens per shard: {args.tokens_per_shard:,}")
    
    # Check for existing checkpoint
    checkpoint_path = output_path / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_tokens = checkpoint["tokens_collected"]
        start_chunk = checkpoint["last_chunk_id"] + 1
        examples_seen = checkpoint.get("examples_seen", 0)
        print(f"  Resuming from checkpoint: {start_tokens:,} tokens, chunk {start_chunk}")
        if examples_seen:
            print(f"  Dataset offset (examples seen): {examples_seen:,}")
    else:
        start_tokens = 0
        start_chunk = 0
        examples_seen = 0
        
        # Save initial metadata
        metadata = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "target_layer": args.target_layer,
            "tokens_per_shard": args.tokens_per_shard,
            "chunk_size": args.chunk_size,
            "model": args.model_name,
            "dataset": args.dataset_name,
            "start_time": str(datetime.now()),
            "max_length": args.max_length,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Starting fresh collection")
    
    # Load model
    print(f"Shard {args.shard_id}: Loading model {args.model_name}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"Shard {args.shard_id}: Model loaded successfully")
        print(f"  Model device: {model.device}")
        print(f"  Number of layers: {len(model.model.layers)}")
        
    except Exception as e:
        print(f"Shard {args.shard_id}: ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Setup hook for activation extraction
    activations_cache = []
    token_ids_cache = []
    
    # Get model's hidden dimension
    d_model = model.config.hidden_size
    print(f"Shard {args.shard_id}: Model hidden size (d_model): {d_model}")
    
    def hook_fn(module, input, output):
        """Hook function to capture layer outputs during forward pass."""
        # Extract activations and move to CPU immediately
        acts = output[0].detach().cpu().half()
        activations_cache.append(acts)
    
    try:
        target_module = model.model.layers[args.target_layer]
        hook_handle = target_module.register_forward_hook(hook_fn)
        print(f"Shard {args.shard_id}: Hook registered on layer {args.target_layer}")
    except Exception as e:
        print(f"Shard {args.shard_id}: ERROR registering hook: {e}")
        sys.exit(1)
    
    # Load dataset
    print(f"Shard {args.shard_id}: Loading dataset {args.dataset_name}...")
    dataset = load_dataset(
        args.dataset_name,
        streaming=True,
        split=args.dataset_split
    )
    
    # Shuffle, shard, and resume from prior offset
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_id)
    if examples_seen > 0:
        dataset = dataset.skip(examples_seen)
    
    # Collection loop
    tokens_collected = start_tokens
    chunk_id = start_chunk
    errors = 0
    
    print(f"Shard {args.shard_id}: Starting collection...")
    print(f"  Target: {args.tokens_per_shard:,} tokens")
    print(f"  Starting from: {tokens_collected:,} tokens")
    print("")
    
    try:
        for example in dataset:
            # Check if we've reached target
            if tokens_collected >= args.tokens_per_shard:
                break
            
            examples_seen += 1
            
            # Tokenize
            try:
                # SlimPajama uses 'text' field
                text_content = example.get("text", example.get("content", ""))
                if not text_content:
                    continue
                
                inputs = tokenizer(
                    text_content,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=False
                ).to(model.device)
                
                if inputs["input_ids"].numel() == 0:
                    continue
                
                # Store token IDs
                token_ids_cache.append(inputs["input_ids"].cpu().numpy())
                
                # Forward pass (activations captured by hook)
                with torch.no_grad():
                    _ = model(**inputs)
                
                tokens_collected += inputs["input_ids"].numel()
                
                # Progress updates every 1M tokens
                if tokens_collected % 1_000_000 < args.max_length:
                    progress_pct = (tokens_collected / args.tokens_per_shard) * 100
                    print(f"Shard {args.shard_id}: {tokens_collected:,} / {args.tokens_per_shard:,} tokens ({progress_pct:.1f}%)")
                
                # Save chunk when threshold reached
                if tokens_collected >= (chunk_id + 1) * args.chunk_size:
                    save_chunk(
                        activations_cache,
                        token_ids_cache,
                        chunk_id,
                        args.output_dir,
                        d_model
                    )
                    
                    # Save checkpoint
                checkpoint = {
                    "shard_id": args.shard_id,
                    "tokens_collected": tokens_collected,
                    "last_chunk_id": chunk_id,
                    "timestamp": str(datetime.now()),
                    "errors": errors,
                    "examples_seen": examples_seen
                }
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, indent=2)
                    
                    print(f"Shard {args.shard_id}: ✓ Saved chunk {chunk_id} (total: {tokens_collected:,} tokens)")
                    
                    # Clear caches to free memory
                    activations_cache.clear()
                    token_ids_cache.clear()
                    chunk_id += 1
                    
            except Exception as e:
                errors += 1
                if errors % 10 == 0:
                    print(f"Shard {args.shard_id}: Warning - {errors} errors encountered (last: {str(e)[:100]})")
                continue
    
    except KeyboardInterrupt:
        print(f"\nShard {args.shard_id}: Interrupted by user")
    except Exception as e:
        print(f"Shard {args.shard_id}: FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final partial chunk if exists
        if activations_cache:
            save_chunk(
                activations_cache,
                token_ids_cache,
                chunk_id,
                args.output_dir,
                d_model
            )
            print(f"Shard {args.shard_id}: ✓ Saved final chunk {chunk_id}")
        
        # Cleanup hook
        hook_handle.remove()
        
        # Final metadata update
        try:
            with open(output_path / "metadata.json", "r") as f:
                metadata = json.load(f)
            metadata["end_time"] = str(datetime.now())
            metadata["tokens_collected"] = tokens_collected
            metadata["chunks_saved"] = chunk_id + 1
            metadata["errors_encountered"] = errors
            metadata["examples_seen"] = examples_seen
            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
        
        print(f"\nShard {args.shard_id}: Collection complete!")
        print(f"  Total tokens: {tokens_collected:,}")
        print(f"  Chunks saved: {chunk_id + 1}")
        print(f"  Errors: {errors}")


def save_chunk(activations_cache, token_ids_cache, chunk_id, output_dir, d_model):
    """Save activations and token IDs to disk."""
    if not activations_cache:
        return
    
    # Concatenate all batches
    all_acts = torch.cat(activations_cache, dim=0)
    all_acts = all_acts.reshape(-1, d_model)  # Flatten to [n_tokens, d_model]
    
    all_toks = np.concatenate(token_ids_cache, axis=0)
    all_toks = all_toks.flatten()
    
    # Ensure same length
    min_len = min(len(all_acts), len(all_toks))
    all_acts = all_acts[:min_len]
    all_toks = all_toks[:min_len]
    
    # Save as numpy arrays
    acts_path = f"{output_dir}/activations/chunk_{chunk_id:04d}.npy"
    toks_path = f"{output_dir}/token_ids/chunk_{chunk_id:04d}.npy"
    
    np.save(acts_path, all_acts.numpy().astype(np.float16))
    np.save(toks_path, all_toks.astype(np.int32))


if __name__ == "__main__":
    main()
