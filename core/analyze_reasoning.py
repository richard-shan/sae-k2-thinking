"""
Example: Using Trained SAE for Reasoning Analysis

This script demonstrates how to use a trained SAE to analyze
reasoning in Kimi-K2-Thinking on specific problems.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.sae_model import load_sae


def analyze_reasoning_problem(model, tokenizer, sae, target_layer, problem_text):
    """
    Analyze which SAE features activate during a reasoning problem.
    
    Args:
        model: K2 model
        tokenizer: K2 tokenizer
        sae: Trained SAE
        target_layer: Layer to analyze
        problem_text: Math/reasoning problem to analyze
    
    Returns:
        dict with analysis results
    """
    print(f"\nAnalyzing: {problem_text}")
    print("-" * 60)
    
    # Hook to capture activations
    activations = []
    def hook(module, input, output):
        activations.append(output[0].detach())
    
    hook_handle = model.model.layers[target_layer].register_forward_hook(hook)
    
    # Tokenize and run model
    inputs = tokenizer(problem_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    hook_handle.remove()
    
    # Get activations and run through SAE
    acts = activations[0].squeeze(0)  # [seq_len, d_model]
    
    with torch.no_grad():
        reconstructed, features = sae(acts)
    
    # Analyze features
    features_np = features.cpu().numpy()
    
    # Per-token analysis
    print("\nPer-Token Feature Analysis:")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    for i, (token, feat_vec) in enumerate(zip(tokens, features_np)):
        active = (feat_vec > 0).sum()
        max_act = feat_vec.max()
        print(f"  Token {i:2d} '{token:15s}': {active:3d} active features, max={max_act:.2f}")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Mean L0 (active features per token): {(features > 0).sum(dim=1).float().mean():.1f}")
    print(f"  Total unique features activated: {(features > 0).any(dim=0).sum()}")
    print(f"  Reconstruction MSE: {((reconstructed - acts)**2).mean():.6f}")
    
    # Top features
    feature_importance = features.sum(dim=0)
    top_features = torch.argsort(feature_importance, descending=True)[:10]
    
    print("\nTop 10 Most Active Features:")
    for rank, feat_id in enumerate(top_features, 1):
        print(f"  {rank:2d}. Feature {feat_id:5d}: total_activation={feature_importance[feat_id]:.2f}")
    
    # Generated response
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"\nModel Response:")
    print(f"  {response}")
    
    return {
        'features': features_np,
        'tokens': tokens,
        'top_features': top_features.cpu().tolist(),
        'response': response,
    }


def main():
    print("=" * 60)
    print("SAE Reasoning Analysis Example")
    print("=" * 60)
    
    # Configuration
    SAE_PATH = "models/sae_layer45_8x/sae_best.pt"
    TARGET_LAYER = 45
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  SAE: {SAE_PATH}")
    print(f"  Target Layer: {TARGET_LAYER}")
    print(f"  Device: {DEVICE}")
    
    # Load models
    print("\nLoading models...")
    
    print("  Loading K2...")
    model = AutoModelForCausalLM.from_pretrained(
        "moonshotai/Kimi-K2-Thinking",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Thinking",
        trust_remote_code=True
    )
    
    print("  Loading SAE...")
    sae = load_sae(SAE_PATH, device=DEVICE)
    
    print("✓ All models loaded")
    
    # Example problems
    problems = [
        "What is 15 × 24?",
        "If x + 7 = 15, what is x?",
        "Prove that the sum of two even numbers is even.",
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
    ]
    
    # Analyze each problem
    results = []
    for problem in problems:
        result = analyze_reasoning_problem(
            model, tokenizer, sae, TARGET_LAYER, problem
        )
        results.append(result)
        print("\n" + "=" * 60)
    
    # Compare features across problems
    print("\nComparing Feature Usage Across Problems:")
    print("-" * 60)
    
    all_features = set()
    for i, result in enumerate(results):
        features_used = set(result['top_features'])
        all_features.update(features_used)
        print(f"\nProblem {i+1}: {problems[i]}")
        print(f"  Top features: {result['top_features'][:5]}")
    
    print(f"\nTotal unique features across all problems: {len(all_features)}")
    
    # Find common features
    common = set(results[0]['top_features'])
    for result in results[1:]:
        common &= set(result['top_features'])
    
    if common:
        print(f"Features common to all problems: {sorted(common)}")
    else:
        print("No features common to all problems (suggests specialization)")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
