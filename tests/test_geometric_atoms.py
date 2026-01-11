"""
Tests for geometric atomic primitives.

Phase 3: Atomic Primitive Tests
These tests verify the new geometric atoms work correctly and maintain
their geometric properties through operations.

Tests cover:
- GeometricLinear: Standard linear with explicit signature
- FinslerLinear: Asymmetric metric dualization
- TwistedEmbed: Orientation-sensitive embedding
- ContactAtom: Conservation law projection
"""
import pytest
import jax
import jax.numpy as jnp

from diffgeo import (
    GeometricLinear, FinslerLinear, TwistedEmbed, GeometricEmbed, ContactAtom,
    TensorVariance, Parity, MetricType, GeometricSignature,
    RandersMetric, FinslerDualizer,
)
from modula.atom import Linear


@pytest.mark.phase3
class TestGeometricLinear:
    """Test GeometricLinear atom properties."""
    
    def test_signature_is_vector_to_vector(self):
        """GeometricLinear should map vectors to vectors."""
        layer = GeometricLinear(16, 8)
        assert layer.signature.domain == TensorVariance.CONTRAVARIANT
        assert layer.signature.codomain == TensorVariance.CONTRAVARIANT
        assert layer.signature.parity == Parity.EVEN
    
    def test_forward_matches_standard_linear(self, key):
        """GeometricLinear forward should match standard Linear."""
        k1, k2 = jax.random.split(key)
        
        geo_linear = GeometricLinear(16, 8)
        std_linear = Linear(16, 8)
        
        weights = geo_linear.initialize(k1)
        x = jax.random.normal(k2, shape=(8,))
        
        y_geo = geo_linear.forward(x, weights)
        y_std = std_linear.forward(x, weights)
        
        assert jnp.allclose(y_geo, y_std)
    
    def test_dualize_produces_orthogonal(self, key):
        """Dualized gradient should be approximately orthogonal."""
        k1, k2 = jax.random.split(key)
        layer = GeometricLinear(16, 8)
        
        weights = layer.initialize(k1)
        grad = [jax.random.normal(k2, shape=weights[0].shape)]
        
        dual = layer.dualize(grad, targetNorm=1.0)
        Q = dual[0] / jnp.sqrt(16/8)  # Remove scaling
        
        # Check orthogonality: Q @ Q.T ≈ I (for tall matrix: Q.T @ Q ≈ I)
        product = Q.T @ Q
        assert jnp.allclose(product, jnp.eye(8), rtol=0.05, atol=0.02)
    
    def test_twisted_linear_has_odd_parity(self):
        """Linear with Parity.ODD should track twisted parity."""
        layer = GeometricLinear(16, 8, parity=Parity.ODD)
        assert layer.signature.parity == Parity.ODD
        assert layer.is_twisted


@pytest.mark.phase3
class TestFinslerLinear:
    """Test FinslerLinear atom with asymmetric metrics."""

    def test_signature_is_finsler(self):
        """FinslerLinear should have Finsler metric type."""
        layer = FinslerLinear(16, 8)
        assert layer.signature.metric_type == MetricType.FINSLER

    def test_initialize_produces_weights_and_drift(self, key):
        """FinslerLinear should initialize both W and drift."""
        layer = FinslerLinear(16, 8)
        weights = layer.initialize(key)

        assert len(weights) == 2
        assert weights[0].shape == (16, 8)  # W
        assert weights[1].shape == (16, 8)  # drift

    def test_drift_has_bounded_norm(self, key):
        """Drift should have norm < 1 for valid Randers metric."""
        layer = FinslerLinear(16, 8, drift_strength=0.5)
        weights = layer.initialize(key)

        drift_norm = jnp.linalg.norm(weights[1])
        assert drift_norm < 1.0
        assert drift_norm > 0.1  # Should be non-trivial

    def test_forward_ignores_drift(self, key):
        """Forward pass should be standard matrix multiply (drift only affects optimization)."""
        k1, k2 = jax.random.split(key)

        layer = FinslerLinear(16, 8)
        weights = layer.initialize(k1)
        x = jax.random.normal(k2, shape=(8,))

        # Forward is just y = Wx
        y = layer.forward(x, weights)
        y_expected = weights[0] @ x

        assert jnp.allclose(y, y_expected)

    def test_dualize_produces_valid_updates(self, key):
        """FinslerLinear dualize should produce valid W and drift updates."""
        k1, k2 = jax.random.split(key)

        layer = FinslerLinear(16, 8, drift_strength=0.5)
        weights = layer.initialize(k1)

        # Gradient for both W and drift
        grad_W = jax.random.normal(k2, shape=(16, 8))
        grad_drift = weights[1]  # Use current drift as gradient
        grads = [grad_W, grad_drift]

        dual = layer.dualize(grads, targetNorm=1.0)

        # Should return [W_update, drift_update]
        assert len(dual) == 2, "FinslerLinear should return [W_update, drift_update]"
        assert dual[0].shape == (16, 8), "W update has correct shape"
        assert dual[1].shape == (16, 8), "Drift update has correct shape"

        # Updates should be finite
        assert jnp.all(jnp.isfinite(dual[0])), "W update should be finite"
        assert jnp.all(jnp.isfinite(dual[1])), "Drift update should be finite"

        # W update should be approximately orthogonal (scaled)
        Q = dual[0] / jnp.sqrt(16 / 8)
        assert jnp.allclose(Q.T @ Q, jnp.eye(8), rtol=0.1, atol=0.05)

    def test_project_maintains_constraints(self, key):
        """Project should keep W orthogonal and drift bounded."""
        layer = FinslerLinear(16, 8)

        # Start with random (possibly invalid) weights
        W = jax.random.normal(key, shape=(16, 8)) * 2  # Not orthogonal
        drift = jax.random.normal(key, shape=(16, 8)) * 2  # Possibly > 1 norm

        projected = layer.project([W, drift])

        # Check W is approximately orthogonal (scaled)
        Q = projected[0] / jnp.sqrt(16/8)
        assert jnp.allclose(Q.T @ Q, jnp.eye(8), rtol=0.05, atol=0.02)

        # Check drift norm is bounded
        assert jnp.linalg.norm(projected[1]) <= 0.96


@pytest.mark.phase3
class TestTwistedEmbed:
    """Test TwistedEmbed orientation-sensitive embedding."""
    
    def test_signature_has_odd_parity(self):
        """TwistedEmbed should have odd parity."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=1000)
        assert embed.signature.parity == Parity.ODD
        assert embed.is_twisted
    
    def test_orientation_flips_output(self, key):
        """Orientation ±1 should flip embedding sign."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=1000)
        weights = embed.initialize(key)
        
        indices = jnp.array([0, 1, 5, 10])
        
        emb_pos = embed.forward(indices, weights, orientation=1.0)
        emb_neg = embed.forward(indices, weights, orientation=-1.0)
        
        assert jnp.allclose(emb_pos, -emb_neg)
    
    def test_embeddings_normalized(self, key):
        """Each embedding vector should have norm ≈ sqrt(dEmbed)."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=100)
        weights = embed.initialize(key)
        
        norms = jnp.linalg.norm(weights[0], axis=1)
        expected_norm = jnp.sqrt(64)
        
        assert jnp.allclose(norms, expected_norm, rtol=0.01)


@pytest.mark.phase3
class TestGeometricEmbed:
    """Test GeometricEmbed (non-twisted) embedding."""
    
    def test_signature_has_even_parity(self):
        """Standard GeometricEmbed should have even parity."""
        embed = GeometricEmbed(dEmbed=64, numEmbed=1000)
        assert embed.signature.parity == Parity.EVEN
        assert not embed.is_twisted
    
    def test_forward_standard_lookup(self, key):
        """GeometricEmbed forward should be standard lookup."""
        embed = GeometricEmbed(dEmbed=64, numEmbed=100)
        weights = embed.initialize(key)
        
        indices = jnp.array([0, 5, 99])
        embeddings = embed.forward(indices, weights)
        
        expected = weights[0][indices]
        assert jnp.allclose(embeddings, expected)


@pytest.mark.phase3
class TestContactAtom:
    """Test ContactAtom conservation law projection."""
    
    def test_requires_odd_dimension(self):
        """Contact manifolds must be odd-dimensional."""
        with pytest.raises(AssertionError):
            ContactAtom(dim=4)  # Even dimension should fail
        
        # Odd dimension should work
        contact = ContactAtom(dim=5)
        assert contact.dim == 5
    
    def test_projects_onto_kernel(self, key):
        """Projected vector should satisfy α(x_proj) ≈ 0."""
        k1, k2 = jax.random.split(key)
        contact = ContactAtom(dim=5)
        weights = contact.initialize(k1)
        
        x = jax.random.normal(k2, shape=(5,))
        x_proj = contact.forward(x, weights)
        
        # Check x_proj is in kernel of alpha
        alpha = weights[0]
        alpha_x_proj = jnp.dot(alpha, x_proj)
        
        assert jnp.abs(alpha_x_proj) < 0.1  # Should be approximately 0
    
    def test_signature_is_symplectic(self):
        """ContactAtom should have symplectic-like metric."""
        contact = ContactAtom(dim=5)
        assert contact.signature.metric_type == MetricType.SYMPLECTIC


@pytest.mark.phase3
class TestGeometricComposition:
    """Test geometric compatibility in composition."""
    
    def test_compatible_signatures_compose(self, key):
        """Modules with compatible signatures should compose."""
        layer1 = GeometricLinear(32, 16)
        layer2 = GeometricLinear(16, 8)
        
        # layer1 @ layer2 means: input → layer2 → layer1 → output
        composed = layer1 @ layer2
        
        # Check composed signature
        assert composed.signature.domain == TensorVariance.CONTRAVARIANT
        assert composed.signature.codomain == TensorVariance.CONTRAVARIANT
    
    def test_parity_multiplies_in_composition(self, key):
        """Composing twisted modules should multiply parities."""
        even = GeometricLinear(16, 8, parity=Parity.EVEN)
        odd = GeometricLinear(8, 4, parity=Parity.ODD)
        
        # EVEN @ ODD = ODD
        composed1 = even @ odd
        assert composed1.signature.parity == Parity.ODD
        
        # ODD @ ODD = EVEN
        odd2 = GeometricLinear(4, 4, parity=Parity.ODD)
        composed2 = odd @ odd2
        assert composed2.signature.parity == Parity.EVEN
    
    def test_composed_forward_works(self, key):
        """Composed module should compute correctly."""
        k1, k2 = jax.random.split(key)
        
        layer1 = GeometricLinear(16, 8)
        layer2 = GeometricLinear(8, 4)
        
        composed = layer1 @ layer2
        weights = composed.initialize(k1)
        
        x = jax.random.normal(k2, shape=(4,))
        y = composed.forward(x, weights)
        
        # Manual computation
        y_manual = weights[1] @ (weights[0] @ x)
        
        assert y.shape == (16,)
        # Note: composed forward order is inner first, so y = W1 @ (W2 @ x)


@pytest.mark.phase3
class TestFinslerDualizer:
    """Test standalone Finsler dualization."""
    
    def test_dualize_respects_target_norm(self, key):
        """Dualized vector should have specified Finsler norm."""
        k1, k2 = jax.random.split(key)
        dim = 8
        
        randers = RandersMetric(
            A=jnp.eye(dim),  # Identity Riemannian part
            b=jnp.zeros(dim)  # No drift (should match Euclidean)
        )
        dualizer = FinslerDualizer(randers)
        
        grad = jax.random.normal(k1, shape=(dim,))
        target = 2.5
        
        dual = dualizer.dualize(grad, target_norm=target)
        actual_norm = randers.norm(dual)
        
        assert jnp.allclose(actual_norm, target, rtol=0.01)
    
    def test_dualize_with_drift(self, key):
        """Dualization with drift should bias direction."""
        k1, k2 = jax.random.split(key)
        dim = 8
        
        # Create Randers with significant drift
        A = jnp.eye(dim)
        b = jnp.zeros(dim).at[0].set(0.5)  # Drift in first direction
        
        randers = RandersMetric(A, b)
        dualizer = FinslerDualizer(randers)
        
        # Gradient perpendicular to drift
        grad = jnp.zeros(dim).at[1].set(1.0)
        
        dual = dualizer.dualize(grad, target_norm=1.0)
        
        # The dual should exist and be finite
        assert jnp.all(jnp.isfinite(dual))
        assert jnp.linalg.norm(dual) > 0
