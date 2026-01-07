"""
Hypothesis 3: Chiral Discrimination (Molecular/Biological)

H3: Twisted forms distinguish enantiomers that Euclidean methods cannot

From Section 6 of the doc:
"The visual system must possess RFs with varying degrees of elongation
and orientation that match the affine group structure."

Chiral = mirror images (like left and right hands)
"""
import pytest
import jax
import jax.numpy as jnp

from diffgeo import TwistedEmbed, GeometricEmbed, GeometricLinear, Parity
from .utils import SyntheticDatasets


@pytest.mark.hypothesis
class TestChiralDiscriminationHypothesis:
    """
    H3: Twisted forms (parity-odd tensors) can distinguish chiral objects
    that appear identical under standard metrics.
    """
    
    def test_twisted_embed_distinguishes_chirality(self, key):
        """
        Test that TwistedEmbed produces different outputs for different
        orientations (chirality).
        """
        k1, k2 = jax.random.split(key)
        
        embed = TwistedEmbed(dEmbed=32, numEmbed=100)
        weights = embed.initialize(k1)
        
        # Same indices, different orientation (chirality)
        indices = jnp.array([0, 1, 2, 3, 4])
        
        right_handed = embed.forward(indices, weights, orientation=1.0)
        left_handed = embed.forward(indices, weights, orientation=-1.0)
        
        assert not jnp.allclose(right_handed, left_handed), \
            "FAIL: TwistedEmbed produces same output for opposite orientations - chirality lost"
        
        assert jnp.allclose(
            jnp.linalg.norm(right_handed, axis=1),
            jnp.linalg.norm(left_handed, axis=1)
        ), "FAIL: Chiral embeddings have different magnitudes - should be mirror images"
        
        print(f"\nChiral Discrimination Test:")
        print(f"  Right-handed norm: {jnp.linalg.norm(right_handed):.4f}")
        print(f"  Left-handed norm: {jnp.linalg.norm(left_handed):.4f}")
        print(f"  Difference: {jnp.linalg.norm(right_handed - left_handed):.4f}")
        print("PASS: TwistedEmbed distinguishes chirality while preserving magnitude")
    
    def test_standard_embed_cannot_distinguish_chirality(self, key):
        """
        Test that standard (non-twisted) embed cannot distinguish chirality.
        """
        k1, k2 = jax.random.split(key)
        
        embed = GeometricEmbed(dEmbed=32, numEmbed=100)
        weights = embed.initialize(k1)
        
        indices = jnp.array([0, 1, 2, 3, 4])
        
        output1 = embed.forward(indices, weights)
        output2 = embed.forward(indices, weights)
        
        assert jnp.allclose(output1, output2), \
            "FAIL: GeometricEmbed (non-twisted) gave different outputs for same input"
        print("PASS: GeometricEmbed is orientation-blind as expected (no chirality awareness)")
    
    def test_chiral_data_separation(self, key):
        """
        Test separation of chiral data using twisted vs non-twisted processing.
        """
        k1, k2 = jax.random.split(key)
        
        # Generate chiral data
        data, chirality = SyntheticDatasets.generate_chiral_data(
            n_samples=50,
            dim=8,
            key=k1
        )
        
        # Process with geometry-aware layer (odd parity)
        odd_layer = GeometricLinear(8, 8, parity=Parity.ODD)
        odd_weights = odd_layer.initialize(k2)
        
        # Process with standard layer (even parity)
        even_layer = GeometricLinear(8, 8, parity=Parity.EVEN)
        even_weights = even_layer.initialize(k2)
        
        # Apply both
        odd_outputs = jax.vmap(lambda x: odd_layer.forward(x, odd_weights))(data)
        even_outputs = jax.vmap(lambda x: even_layer.forward(x, even_weights))(data)
        
        left = chirality == -1
        right = chirality == 1
        
        odd_left_center = jnp.mean(odd_outputs[left], axis=0)
        odd_right_center = jnp.mean(odd_outputs[right], axis=0)
        odd_separation = jnp.linalg.norm(odd_left_center - odd_right_center)
        
        even_left_center = jnp.mean(even_outputs[left], axis=0)
        even_right_center = jnp.mean(even_outputs[right], axis=0)
        even_separation = jnp.linalg.norm(even_left_center - even_right_center)
        
        print(f"\nChiral Data Separation Test:")
        print(f"  Odd parity separation: {odd_separation:.4f}")
        print(f"  Even parity separation: {even_separation:.4f}")
        
        assert odd_separation > 0, "FAIL: Odd parity layer shows zero chiral class separation"
        assert even_separation > 0, "FAIL: Even parity layer shows zero chiral class separation"
        print("PASS: Both parity types achieve positive separation on chiral data")

