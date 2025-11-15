"""
Sparse Autoencoder Model Definition

Implements a sparse autoencoder for learning interpretable features from
neural network activations. Based on SAELens architecture (Neel Nanda).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for interpreting model activations.
    
    Architecture (following SAELens best practices):
        1. Optional input normalization (normalize_activations)
        2. Optional b_dec applied to input (apply_b_dec_to_input)
        3. encoder: d_model -> d_sae (with bias)
        4. ReLU activation (induces sparsity)
        5. decoder: d_sae -> d_model (with bias b_dec)
    
    The decoder is constrained to have unit-norm columns for interpretability.
    
    Args:
        d_model (int): Input/output dimension (e.g., 7168 for K2 layer 45)
        d_sae (int): SAE hidden dimension (typically 4-16× d_model)
        normalize_activations (str): Normalization strategy
            - None: No normalization
            - "expected_average_only_in": Center based on expected mean
            - "layer_norm": Apply layer normalization
        apply_b_dec_to_input (bool): Whether to apply decoder bias to input
    """
    
    def __init__(
        self, 
        d_model: int = 7168, 
        d_sae: int = 57344,
        normalize_activations: str = "expected_average_only_in",
        apply_b_dec_to_input: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.normalize_activations = normalize_activations
        self.apply_b_dec_to_input = apply_b_dec_to_input
        
        # Encoder: maps (centered) activations to feature space
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder: reconstructs activations from features
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Activation statistics for normalization
        if normalize_activations == "expected_average_only_in":
            self.register_buffer(
                "activation_mean", 
                torch.zeros(d_model)
            )
            self.register_buffer(
                "activation_mean_initialized",
                torch.tensor(False)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following SAELens conventions."""
        # Kaiming uniform for encoder
        nn.init.kaiming_uniform_(self.W_enc, a=0, mode='fan_in', nonlinearity='relu')
        
        # Decoder initialized as normalized encoder transpose
        self.W_dec.data = self.W_enc.data.t().clone()
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
        
        # Biases initialized to zero (already done in __init__)
    
    def encode(self, x):
        """
        Encode activations to sparse features.
        
        Args:
            x: Input activations [batch, d_model]
        
        Returns:
            feature_acts: Sparse feature activations [batch, d_sae]
        """
        # Center activations if using normalization
        x_centered = self._center_activations(x)
        
        # Apply decoder bias to input if configured
        if self.apply_b_dec_to_input:
            x_centered = x_centered - self.b_dec
        
        # Encode: (x_centered @ W_enc.T) + b_enc
        pre_acts = F.linear(x_centered, self.W_enc, self.b_enc)
        
        # Apply ReLU for sparsity
        feature_acts = F.relu(pre_acts)
        
        return feature_acts
    
    def decode(self, feature_acts):
        """
        Decode sparse features back to activations.
        
        Args:
            feature_acts: Sparse feature activations [batch, d_sae]
        
        Returns:
            reconstructed: Reconstructed activations [batch, d_model]
        """
        # Decode: (feature_acts @ W_dec.T) + b_dec
        reconstructed = F.linear(feature_acts, self.W_dec, self.b_dec)
        return reconstructed
    
    def _center_activations(self, x):
        """Center activations based on normalization strategy."""
        if self.normalize_activations == "expected_average_only_in":
            return x - self.activation_mean
        elif self.normalize_activations == "layer_norm":
            return F.layer_norm(x, (self.d_model,))
        else:
            return x
    
    def update_activation_mean(self, x):
        """
        Update running estimate of activation mean.
        Should be called during training on each batch.
        """
        if self.normalize_activations == "expected_average_only_in":
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                if not self.activation_mean_initialized:
                    self.activation_mean.copy_(batch_mean)
                    self.activation_mean_initialized.fill_(True)
                else:
                    # Running average with momentum 0.01
                    momentum = 0.01
                    self.activation_mean.mul_(1 - momentum).add_(batch_mean * momentum)
    
    def forward(self, x):
        """
        Forward pass through SAE.
        
        Args:
            x: Input activations [batch, d_model]
        
        Returns:
            reconstructed: Reconstructed activations [batch, d_model]
            feature_acts: Sparse feature activations [batch, d_sae]
            loss_dict: Dictionary with auxiliary losses
        """
        # Update activation statistics if in training mode
        if self.training and self.normalize_activations == "expected_average_only_in":
            self.update_activation_mean(x)
        
        # Encode to sparse features
        feature_acts = self.encode(x)
        
        # Decode back to activation space
        reconstructed = self.decode(feature_acts)
        
        # Auxiliary loss dictionary (for training)
        loss_dict = {}
        
        return reconstructed, feature_acts, loss_dict
    
    def normalize_decoder(self):
        """
        Normalize decoder columns to unit norm.
        
        This should be called after each optimizer step to maintain
        the constraint that decoder columns have unit norm.
        """
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
    
    def get_feature_norms(self):
        """
        Get L2 norms of decoder columns (should be ~1.0 if normalized).
        
        Returns:
            norms: Tensor of shape [d_sae] with L2 norm of each feature
        """
        return torch.norm(self.W_dec.data, dim=0)
    
    def get_sparsity_metrics(self, feature_acts):
        """
        Compute sparsity metrics for feature activations.
        
        Args:
            feature_acts: Feature activations [batch, d_sae]
        
        Returns:
            dict with:
                - l0: Average number of active features per example
                - l1: Average L1 norm of features
                - frac_active: Fraction of features that are active
        """
        l0 = (feature_acts > 0).float().sum(dim=1).mean().item()
        l1 = feature_acts.abs().mean().item()
        frac_active = (feature_acts > 0).any(dim=0).float().mean().item()
        
        return {
            "l0": l0,
            "l1": l1,
            "frac_active": frac_active
        }


def load_sae(checkpoint_path, device='cuda'):
    """
    Load a trained SAE from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on
    
    Returns:
        sae: Loaded SAE model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    sae = SparseAutoencoder(
        d_model=checkpoint['d_model'],
        d_sae=checkpoint['d_sae']
    ).to(device)
    
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    return sae


def save_sae(sae, save_path, metadata=None):
    """
    Save SAE model with metadata.
    
    Args:
        sae: SAE model to save
        save_path: Path to save checkpoint
        metadata: Optional dict of metadata to include
    """
    checkpoint = {
        'model_state_dict': sae.state_dict(),
        'd_model': sae.d_model,
        'd_sae': sae.d_sae,
    }
    
    if metadata is not None:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, save_path)
