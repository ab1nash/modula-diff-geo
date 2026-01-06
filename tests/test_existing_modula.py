"""
Regression tests for existing modula functionality.

Ensures geometric extensions don't break base modula behavior.
These tests verify the core modula abstractions work correctly.
"""
import pytest
import jax
import jax.numpy as jnp

from modula.abstract import Module, Atom, Bond, CompositeModule, Identity
from modula.atom import Linear, Embed, orthogonalize
from modula.bond import ReLU, GeLU


class TestModulaAtoms:
    """Test base atom functionality."""
    
    def test_linear_forward(self, key):
        """Linear layer computes Wx correctly."""
        linear = Linear(8, 4)
        weights = linear.initialize(key)
        x = jnp.ones(4)
        y = linear.forward(x, weights)
        assert y.shape == (8,)
    
    def test_linear_initialize_shape(self, key):
        """Linear initialization produces correct shape."""
        linear = Linear(16, 8)
        weights = linear.initialize(key)
        assert len(weights) == 1
        assert weights[0].shape == (16, 8)
    
    def test_linear_project_maintains_shape(self, key):
        """Projection doesn't change weight shape."""
        linear = Linear(8, 8)
        weights = linear.initialize(key)
        projected = linear.project(weights)
        assert weights[0].shape == projected[0].shape
    
    def test_embed_forward(self, key):
        """Embed looks up vectors by index."""
        embed = Embed(16, 100)  # 100 tokens, dim 16
        weights = embed.initialize(key)
        indices = jnp.array([0, 5, 10])
        output = embed.forward(indices, weights)
        assert output.shape == (3, 16)
    
    def test_embed_initialization_normalized(self, key):
        """Embed rows should be normalized."""
        embed = Embed(8, 50)
        weights = embed.initialize(key)
        norms = jnp.linalg.norm(weights[0], axis=1)
        expected = jnp.sqrt(8) * jnp.ones(50)
        assert jnp.allclose(norms, expected, rtol=1e-5)


class TestModulaBonds:
    """Test bond (activation) functionality."""
    
    def test_relu_forward(self):
        """ReLU zeros negative values."""
        relu = ReLU()
        x = jnp.array([-1.0, 0.0, 1.0])
        y = relu.forward(x, [])
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(y, expected)
    
    def test_gelu_forward(self):
        """GeLU is smooth approximation of ReLU."""
        gelu = GeLU()
        x = jnp.array([-1.0, 0.0, 1.0])
        y = gelu.forward(x, [])
        # GeLU(-1) ≈ -0.159, GeLU(0) = 0, GeLU(1) ≈ 0.841
        assert y[1] == 0.0
        assert y[0] < 0  # Negative but not zero
        assert y[2] > 0
    
    def test_identity_passthrough(self):
        """Identity returns input unchanged."""
        identity = Identity()
        x = jnp.array([1.0, 2.0, 3.0])
        y = identity.forward(x, [])
        assert jnp.array_equal(x, y)


class TestModulaComposition:
    """Test module composition."""
    
    def test_composition_atoms_count(self, key):
        """Composed module has sum of atoms."""
        l1 = Linear(8, 4)
        l2 = Linear(4, 8)
        composed = l2 @ l1
        assert composed.atoms == 2
    
    def test_composition_forward(self, key):
        """Composed forward chains correctly."""
        l1 = Linear(8, 4)
        relu = ReLU()
        l2 = Linear(4, 8)
        
        model = l2 @ relu @ l1
        weights = model.initialize(key)
        x = jnp.ones(4)
        y = model.forward(x, weights)
        assert y.shape == (4,)
    
    def test_composition_sensitivity(self, key):
        """Sensitivity multiplies through composition."""
        l1 = Linear(8, 4)
        l2 = Linear(4, 8)
        composed = l2 @ l1
        assert composed.sensitivity == l1.sensitivity * l2.sensitivity
    
    def test_composition_mass_adds(self, key):
        """Mass adds through composition."""
        l1 = Linear(8, 4)
        l2 = Linear(4, 8)
        composed = l2 @ l1
        assert composed.mass == l1.mass + l2.mass


class TestOrthogonalize:
    """Test Newton-Schulz orthogonalization."""
    
    def test_orthogonalize_square(self):
        """Orthogonalize works on square matrices."""
        key = jax.random.PRNGKey(123)  # Fixed seed for reproducibility
        A = jax.random.normal(key, shape=(8, 8), dtype=jnp.float32)
        Q = orthogonalize(A)
        assert jnp.allclose(Q @ Q.T, jnp.eye(8), atol=0.01)
    
    def test_orthogonalize_tall(self):
        """Orthogonalize works on tall matrices."""
        key = jax.random.PRNGKey(456)
        A = jax.random.normal(key, shape=(16, 8), dtype=jnp.float32)
        Q = orthogonalize(A)
        assert jnp.allclose(Q.T @ Q, jnp.eye(8), atol=0.01)
    
    def test_orthogonalize_wide(self):
        """Orthogonalize works on wide matrices."""
        key = jax.random.PRNGKey(789)
        A = jax.random.normal(key, shape=(8, 16), dtype=jnp.float32)
        Q = orthogonalize(A)
        assert jnp.allclose(Q @ Q.T, jnp.eye(8), atol=0.01)


class TestDualization:
    """Test dualization (gradient to update mapping)."""
    
    def test_linear_dualize_shape(self, key):
        """Dualize produces same shape as gradient."""
        k1, k2 = jax.random.split(key)
        linear = Linear(8, 4)
        weights = linear.initialize(k1)
        grad = [jax.random.normal(k2, shape=weights[0].shape)]
        dual = linear.dualize(grad, targetNorm=1.0)
        assert dual[0].shape == grad[0].shape
    
    def test_composite_dualize(self, key):
        """Composite module dualizes all atoms."""
        k1, k2 = jax.random.split(key)
        l1 = Linear(8, 4)
        l2 = Linear(4, 8)
        model = l2 @ l1
        
        weights = model.initialize(k1)
        grads = [jax.random.normal(k2, shape=w.shape) for w in weights]
        duals = model.dualize(grads, targetNorm=1.0)
        
        assert len(duals) == len(grads)
        for d, g in zip(duals, grads):
            assert d.shape == g.shape

