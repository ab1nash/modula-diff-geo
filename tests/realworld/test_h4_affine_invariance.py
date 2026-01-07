"""
Hypothesis 4: Affine Invariant Recognition (Vision)

H4: Geometric covariance provides robustness to perspective distortion

From Section 6.1 of the doc:
"When an object is viewed from a slant, its image undergoes an affine
transformation (foreshortening). To maintain a stable representation,
the visual system must possess RFs with varying degrees of elongation."
"""
import pytest
import jax
import jax.numpy as jnp

from diffgeo import GeometricLinear, MetricTensor
from .utils import SyntheticDatasets


@pytest.mark.hypothesis
class TestAffineInvarianceHypothesis:
    """
    H4: Geometric covariance provides robustness to perspective distortion.
    """
    
    def test_metric_invariance_under_affine(self, key):
        """
        Test that metric tensor correctly transforms under affine changes.
        
        From Section 4.2:
        "This metric is invariant under affine transformations of the data."
        """
        k1, k2 = jax.random.split(key)
        
        dim = 4
        
        # Create a metric tensor
        L = jax.random.normal(k1, (dim, dim))
        metric_matrix = L @ L.T + 0.1 * jnp.eye(dim)
        metric = MetricTensor(metric_matrix)
        
        # Create two vectors
        v1 = jax.random.normal(k2, (dim,))
        v2 = jax.random.normal(jax.random.split(k2)[0], (dim,))
        
        # Original inner product
        original_inner = metric.inner_product(v1, v2)
        
        # Apply random affine transformation
        W = jax.random.normal(jax.random.split(k2)[1], (dim, dim))
        W = W + 0.5 * jnp.eye(dim)  # Ensure invertible
        
        # Transform vectors
        v1_transformed = W @ v1
        v2_transformed = W @ v2
        
        # Transform metric: g' = W^{-T} g W^{-1}
        W_inv = jnp.linalg.inv(W)
        transformed_metric_matrix = W_inv.T @ metric_matrix @ W_inv
        transformed_metric = MetricTensor(transformed_metric_matrix)
        
        # Inner product should be preserved
        transformed_inner = transformed_metric.inner_product(v1_transformed, v2_transformed)
        
        print(f"\nAffine Invariance Test:")
        print(f"  Original inner product: {original_inner:.6f}")
        print(f"  Transformed inner product: {transformed_inner:.6f}")
        
        assert jnp.allclose(original_inner, transformed_inner, rtol=1e-4), \
            "FAIL: Inner product not preserved under affine transform - metric transformation incorrect"
        print("PASS: Metric tensor correctly transforms, preserving inner products under affine maps")
    
    def test_pattern_recognition_under_distortion(self, key):
        """
        Test that geometric processing maintains pattern identity under
        affine distortion.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        dim = 8
        
        # Create a "canonical" pattern
        base_pattern = jax.random.normal(k1, (dim,))
        base_pattern = base_pattern / jnp.linalg.norm(base_pattern)
        
        # Generate affine-transformed versions
        transformed, transforms = SyntheticDatasets.generate_affine_transformed_data(
            n_samples=20,
            base_pattern=base_pattern,
            key=k2,
            transform_strength=0.3
        )
        
        # Generate a different pattern
        other_pattern = jax.random.normal(k3, (dim,))
        other_pattern = other_pattern / jnp.linalg.norm(other_pattern)
        
        # Process all through a geometric layer
        layer = GeometricLinear(dim, dim)
        weights = layer.initialize(k3)
        
        processed_base = layer.forward(base_pattern, weights)
        processed_transformed = jax.vmap(lambda x: layer.forward(x, weights))(transformed)
        processed_other = layer.forward(other_pattern, weights)
        
        # Check: transformed versions should be closer to base than to other
        dist_to_base = jax.vmap(lambda x: jnp.linalg.norm(x - processed_base))(processed_transformed)
        dist_to_other = jax.vmap(lambda x: jnp.linalg.norm(x - processed_other))(processed_transformed)
        
        closer_to_base = jnp.sum(dist_to_base < dist_to_other)
        
        print(f"\nPattern Recognition Under Distortion:")
        print(f"  Mean distance to base: {jnp.mean(dist_to_base):.4f}")
        print(f"  Mean distance to other: {jnp.mean(dist_to_other):.4f}")
        print(f"  Samples closer to base: {closer_to_base}/20")
        
        assert closer_to_base >= 10, \
            f"FAIL: Only {closer_to_base}/20 closer to base - affine distortion breaks recognition"
        print("PASS: Geometric layer maintains pattern identity under affine distortion")

