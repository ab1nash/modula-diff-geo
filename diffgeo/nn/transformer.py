"""
Geometric Transformer: Finslerian extensions to Modula's attention.

This module provides geometric versions of Modula's attention components
using existing geometric atoms (FinslerLinear, TwistedEmbed, ContactAtom).

The key insight is that geometric covariance comes from the *weight space*
geometry (FinslerLinear), not from modifying the attention computation itself.
Standard AttentionQK is reused as-is.

Components:
- GeometricAttention(): Attention with FinslerLinear projections
- GeometricGPT(): Full transformer with TwistedEmbed + FinslerLinear

The asymmetry/directionality comes from:
1. FinslerLinear's drift vectors in Q, K, V, O projections
2. TwistedEmbed's orientation sensitivity
3. Standard causal masking

Reference: "Geometric Covariance in Deep Sequence Modeling" Report
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from modula.abstract import Identity
from modula.bond import (
    SplitIntoHeads, MergeHeads, AttentionQK, 
    CausalMask, Softmax, ApplyAttentionScores, GeLU
)

from .atoms import FinslerLinear, TwistedEmbed
from .bonds import RopeJIT


# =============================================================================
# Geometric Attention (using standard AttentionQK)
# =============================================================================

def GeometricAttention(num_heads, d_embed, d_query, d_value, 
                       attention_scale=1.0, drift_strength=0.3):
    """
    Multi-head attention with Finsler geometry in projections.
    
    Uses standard Modula attention (AttentionQK, CausalMask, Softmax) but
    replaces Linear with FinslerLinear for Q, K, V, O projections.
    
    The geometric properties come from FinslerLinear:
    - Asymmetric weight updates via drift vectors
    - Directed pattern learning (causal structure in weight space)
    
    This is mathematically equivalent to standard attention in the forward
    pass, but differs in how gradients are transformed during training.
    
    Args:
        num_heads: Number of attention heads
        d_embed: Embedding dimension
        d_query: Query/Key dimension per head
        d_value: Value dimension per head
        attention_scale: Softmax temperature
        drift_strength: FinslerLinear drift magnitude (0 = Euclidean)
        
    Returns:
        Composable attention module
        
    Example:
        att = GeometricAttention(4, 128, 32, 32)
        y = att(x, weights)
    """
    # Q, K, V projections with Finsler geometry (asymmetric weight updates)
    Q = SplitIntoHeads(num_heads) @ FinslerLinear(num_heads * d_query, d_embed, drift_strength)
    K = SplitIntoHeads(num_heads) @ FinslerLinear(num_heads * d_query, d_embed, drift_strength)
    V = SplitIntoHeads(num_heads) @ FinslerLinear(num_heads * d_value, d_embed, drift_strength)
    W = FinslerLinear(d_embed, num_heads * d_value, drift_strength) @ MergeHeads()
    
    # Standard attention: RoPE → QK → CausalMask → Softmax
    # Uses RopeJIT (JIT-compatible, no Python caching that breaks tracing)
    AttentionScores = (
        Softmax(attention_scale) @ 
        CausalMask() @ 
        AttentionQK() @ 
        RopeJIT(d_query) @ 
        (Q, K)
    )
    
    # Apply attention and project back
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)


# =============================================================================
# Geometric GPT
# =============================================================================

def GeometricGPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks,
                 blocks_mass=5, attention_scale=1.0, final_scale=1.0,
                 drift_strength=0.3, orientation=1.0):
    """
    GPT with geometric covariance via FinslerLinear and TwistedEmbed.
    
    Differences from standard Modula GPT:
    1. TwistedEmbed: Orientation-sensitive embeddings (chirality)
    2. FinslerLinear: Asymmetric weight updates in all projections
    
    The orientation parameter controls chirality:
    - orientation=+1: Standard ("right-handed") processing
    - orientation=-1: Reflected ("left-handed") processing
    
    For training:
    - Use orientation=+1 for standard causal LM
    - Sample orientation ∈ {+1, -1} for chirality-invariant learning
    - Estimate orientation from data for downstream tasks
    
    Args:
        vocab_size: Vocabulary size
        num_heads: Attention heads per block
        d_embed: Model dimension
        d_query: Query/Key dimension per head
        d_value: Value dimension per head
        num_blocks: Number of transformer blocks
        blocks_mass: Mass for feature learning allocation
        attention_scale: Softmax temperature
        final_scale: Output projection scale
        drift_strength: Finsler drift for asymmetric weight updates
        orientation: Chirality orientation (+1 or -1)
        
    Returns:
        GeometricGPT module (composable)
        
    Example:
        model = GeometricGPT(vocab_size=65, num_heads=4, d_embed=128, 
                             d_query=32, d_value=32, num_blocks=4)
        weights = model.initialize(key)
        logits = model(tokens, weights)
    """
    # Twisted embedding (orientation-sensitive)
    embed = TwistedEmbedWrapper(d_embed, vocab_size, orientation)
    embed.tare()
    
    # Geometric attention and MLP (both use FinslerLinear)
    att = GeometricAttention(num_heads, d_embed, d_query, d_value, 
                             attention_scale, drift_strength)
    mlp = (FinslerLinear(d_embed, 4*d_embed, drift_strength) @ 
           GeLU() @ 
           FinslerLinear(4*d_embed, d_embed, drift_strength))
    
    # Standard Modula residual pattern
    att_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    
    # Stack blocks
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)
    
    # Output projection
    out = final_scale * FinslerLinear(vocab_size, d_embed, drift_strength)
    
    return out @ blocks @ embed


class TwistedEmbedWrapper(TwistedEmbed):
    """
    Wrapper to pass orientation through standard Modula interface.
    
    The orientation is stored at construction time and applied
    during forward pass. For dynamic orientation, create a new
    wrapper or use TwistedEmbed directly with explicit orientation.
    """
    
    def __init__(self, d_embed, vocab_size, orientation=1.0):
        super().__init__(d_embed, vocab_size)
        self.orientation = orientation
    
    def forward(self, inputData, weightsList):
        return super().forward(inputData, weightsList, orientation=self.orientation)


# =============================================================================
# Factory functions
# =============================================================================

def create_geometric_gpt(vocab_size, num_heads=4, d_embed=128, num_blocks=4,
                         drift_strength=0.3, orientation=1.0):
    """
    Create a standard Geometric GPT with sensible defaults.
    
    Args:
        vocab_size: Size of vocabulary
        num_heads: Attention heads (default 4)
        d_embed: Model dimension (default 128)
        num_blocks: Number of layers (default 4)
        drift_strength: Finsler asymmetry (default 0.3)
        orientation: Chirality (default +1)
        
    Returns:
        GeometricGPT model
    """
    d_query = d_embed // num_heads
    d_value = d_embed // num_heads
    
    return GeometricGPT(
        vocab_size=vocab_size,
        num_heads=num_heads,
        d_embed=d_embed,
        d_query=d_query,
        d_value=d_value,
        num_blocks=num_blocks,
        drift_strength=drift_strength,
        orientation=orientation,
    )


def create_chiral_pair(vocab_size, num_heads=4, d_embed=128, num_blocks=4,
                       drift_strength=0.3):
    """
    Create a pair of Geometric GPTs for chirality discrimination.
    
    Returns models with opposite orientations for contrastive learning
    or chirality-invariant training.
    
    Args:
        vocab_size: Size of vocabulary
        num_heads: Attention heads
        d_embed: Model dimension
        num_blocks: Number of layers
        drift_strength: Finsler asymmetry
        
    Returns:
        Tuple of (left_handed_model, right_handed_model)
    """
    d_query = d_embed // num_heads
    d_value = d_embed // num_heads
    
    left = GeometricGPT(
        vocab_size, num_heads, d_embed, d_query, d_value, num_blocks,
        drift_strength=drift_strength, orientation=+1.0
    )
    
    right = GeometricGPT(
        vocab_size, num_heads, d_embed, d_query, d_value, num_blocks,
        drift_strength=drift_strength, orientation=-1.0
    )
    
    return left, right


# =============================================================================
# JIT-Compatible Standard GPT (for fair benchmarking)
# =============================================================================

def StandardGPTJIT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks,
                   blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    """
    Standard GPT using RopeJIT for JIT compatibility.
    
    This is identical to modula's GPT() but uses RopeJIT instead of Rope,
    enabling full JIT compilation without tracer leaks.
    
    Use this as a fair baseline when comparing against GeometricGPT,
    since both use the same JIT-compatible infrastructure.
    
    Args:
        vocab_size: Size of token vocabulary
        num_heads: Number of attention heads
        d_embed: Embedding dimension  
        d_query: Query dimension per head
        d_value: Value dimension per head
        num_blocks: Number of transformer blocks
        blocks_mass: Mass for modular normalization
        attention_scale: Softmax temperature
        final_scale: Output scaling
        
    Returns:
        JIT-compatible GPT model using modula composition
    """
    from modula.atom import Linear, Embed
    from modula.bond import (
        SplitIntoHeads, MergeHeads, AttentionQK, 
        CausalMask, Softmax, ApplyAttentionScores, GeLU
    )
    
    d_head = d_embed // num_heads
    
    # Attention with RopeJIT instead of Rope
    def Attention():
        Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
        K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
        V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
        W = Linear(d_embed, num_heads * d_value) @ MergeHeads()
        
        AttentionScores = (
            Softmax(attention_scale) @ 
            CausalMask() @ 
            AttentionQK() @ 
            RopeJIT(d_query) @  # JIT-compatible Rope
            (Q, K)
        )
        return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)
    
    # Standard components
    embed = Embed(d_embed, vocab_size)
    out = final_scale * Linear(vocab_size, d_embed)
    
    att = Attention()
    mlp = Linear(d_embed, 4*d_embed) @ GeLU() @ Linear(4*d_embed, d_embed)
    
    att_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.mass = blocks_mass
    
    return out @ blocks @ embed


__all__ = [
    # Building blocks
    'GeometricAttention',
    # Full models
    'GeometricGPT',
    'StandardGPTJIT',
    'TwistedEmbedWrapper',
    # Factory functions
    'create_geometric_gpt',
    'create_chiral_pair',
]
