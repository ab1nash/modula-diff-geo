"""
Integration Tests: Full Geometric Pipeline

Tests combining multiple geometric components to verify end-to-end functionality.
"""
import pytest
import jax
import jax.numpy as jnp

from diffgeo import GeometricLinear


@pytest.mark.hypothesis
class TestFullGeometricPipeline:
    """Integration test combining multiple geometric components."""
    
    def test_geometric_pipeline_gradient_flow(self, key):
        """Test that gradients flow correctly through geometric pipeline."""
        k1, k2 = jax.random.split(key)
        
        # Build pipeline: Linear → Linear
        layer1 = GeometricLinear(16, 32)
        layer2 = GeometricLinear(32, 16)
        
        net = layer1 @ layer2
        weights = net.initialize(k1)
        
        # Create batch of inputs
        batch = jax.random.normal(k2, (10, 16))
        target = jnp.ones((10, 16))
        
        def loss_fn(w):
            outputs = jax.vmap(lambda x: net.forward(x, w))(batch)
            return jnp.mean((outputs - target) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(weights)
        
        print(f"\nFull Pipeline Gradient Flow Test:")
        print(f"  Initial loss: {loss:.4f}")
        print(f"  Number of gradient tensors: {len(grads)}")
        
        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"FAIL: Gradient {i} has NaN/Inf - gradient flow is broken"
            print(f"  Gradient {i} norm: {jnp.linalg.norm(g):.4f}")
        print("PASS: All gradients finite - geometric pipeline maintains stable gradient flow")

