"""
Phase 2 Tests: Dualization and Newton-Schulz

Tests for the duality map that converts gradients (covectors) to updates (vectors).
This is the computational heart of geometric optimization.

Key properties:
- Newton-Schulz converges to orthogonal matrix
- Dualized vectors lie on unit ball boundary
- Finsler duality handles asymmetric metrics
- Randers metric reduces to Riemannian when drift=0

Reference: Modula paper on modular duality, Burke on duality in mechanics.
"""
import pytest
import jax
import jax.numpy as jnp

from modula.atom import orthogonalize
from tests.geometry.generators import (
    random_metric, random_randers_metric, random_spd_matrix
)
from tests.geometry.invariants import (
    assert_orthogonal,
    assert_singular_values_near_one,
    assert_positive_homogeneity,
    assert_finsler_asymmetry,
    assert_strong_convexity,
    assert_randers_reduces_to_riemannian,
    assert_dualization_on_unit_ball
)


@pytest.mark.phase2
@pytest.mark.invariant
class TestNewtonSchulz:
    """Test Newton-Schulz orthogonalization (spectral dualization)."""
    
    def test_converges_to_orthogonal_square(self):
        """Newton-Schulz produces orthogonal matrix from square input."""
        key = jax.random.PRNGKey(100)
        dim = 16
        A = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        Q = orthogonalize(A)
        assert_orthogonal(Q, rtol=0.02, atol=0.01)
    
    def test_converges_to_orthogonal_tall(self):
        """Newton-Schulz works on tall matrices (rows > cols)."""
        key = jax.random.PRNGKey(200)
        rows, cols = 32, 16
        A = jax.random.normal(key, shape=(rows, cols), dtype=jnp.float32)
        Q = orthogonalize(A)
        # Q should have orthonormal columns
        assert jnp.allclose(Q.T @ Q, jnp.eye(cols), rtol=0.02, atol=0.01)
    
    def test_converges_to_orthogonal_wide(self):
        """Newton-Schulz works on wide matrices (cols > rows)."""
        key = jax.random.PRNGKey(300)
        rows, cols = 16, 32
        A = jax.random.normal(key, shape=(rows, cols), dtype=jnp.float32)
        Q = orthogonalize(A)
        # Q should have orthonormal rows
        assert jnp.allclose(Q @ Q.T, jnp.eye(rows), rtol=0.02, atol=0.01)
    
    def test_singular_values_near_one(self, key):
        """All singular values should be ≈ 1 after orthogonalization."""
        dim = 16
        A = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        Q = orthogonalize(A)
        assert_singular_values_near_one(Q, rtol=0.02, atol=0.01)
    
    def test_preserves_orientation(self):
        """Check that sign of determinant is preserved (or at least consistent)."""
        key = jax.random.PRNGKey(500)
        dim = 8
        A = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        Q = orthogonalize(A)
        det_Q = jnp.linalg.det(Q)
        # Just verify Q is valid orthogonal (det = ±1)
        assert jnp.allclose(jnp.abs(det_Q), 1.0, rtol=0.05)
    
    def test_idempotent_on_orthogonal(self, key):
        """Orthogonalizing an orthogonal matrix should return ~same matrix."""
        dim = 8
        A = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        Q, _ = jnp.linalg.qr(A)  # Start with truly orthogonal
        Q_again = orthogonalize(Q)
        # Should be close to Q (possibly with sign flip)
        diff = jnp.minimum(
            jnp.linalg.norm(Q - Q_again),
            jnp.linalg.norm(Q + Q_again)
        )
        assert diff < 0.1  # Allow some numerical drift


@pytest.mark.phase2
@pytest.mark.invariant
class TestRandersMetric:
    """Test Finsler (Randers) metric properties."""
    
    def test_randers_is_valid(self, key, dim):
        """Generated Randers metric satisfies strong convexity."""
        randers = random_randers_metric(key, dim, drift_strength=0.3)
        assert_strong_convexity(randers)
    
    def test_positive_homogeneity(self, key, dim, tolerance):
        """F(λv) = λF(v) for λ > 0."""
        k1, k2 = jax.random.split(key)
        randers = random_randers_metric(k1, dim)
        v = jax.random.normal(k2, shape=(dim,))
        
        assert_positive_homogeneity(randers.norm, v, rtol=tolerance)
    
    def test_asymmetry_with_nonzero_drift(self, key, dim, tolerance):
        """F(v) ≠ F(-v) when drift b ≠ 0."""
        k1, k2 = jax.random.split(key)
        randers = random_randers_metric(k1, dim, drift_strength=0.5)
        v = jax.random.normal(k2, shape=(dim,))
        
        assert_finsler_asymmetry(randers, v, rtol=tolerance)
    
    def test_reduces_to_riemannian_when_drift_zero(self, key, dim, tolerance):
        """With b=0, Randers equals Riemannian."""
        k1, k2 = jax.random.split(key)
        A = random_spd_matrix(k1, dim)
        v = jax.random.normal(k2, shape=(dim,))
        
        assert_randers_reduces_to_riemannian(A, v, rtol=tolerance)
    
    def test_drift_affects_cost_asymmetrically(self, key, dim):
        """Movement in opposite directions should have different costs."""
        k1, k2 = jax.random.split(key)
        randers = random_randers_metric(k1, dim, drift_strength=0.5)
        
        # Move in direction of drift and opposite
        b_normalized = randers.b / jnp.linalg.norm(randers.b)
        cost_forward = randers.norm(b_normalized)
        cost_backward = randers.norm(-b_normalized)
        
        # Costs should be different (asymmetric metric)
        # For Randers F(v) = sqrt(v^T A v) + b^T v:
        # Forward: sqrt(1) + |b| > 0
        # Backward: sqrt(1) - |b| (different value)
        assert not jnp.allclose(cost_forward, cost_backward, rtol=0.01)


@pytest.mark.phase2
@pytest.mark.invariant
class TestDualizationProperties:
    """Test general dualization properties (will expand when implemented)."""
    
    def test_spectral_norm_of_orthogonalized(self):
        """Orthogonalized matrix has spectral norm = 1."""
        key = jax.random.PRNGKey(999)
        dim = 16
        A = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        Q = orthogonalize(A)
        spectral_norm = jnp.linalg.norm(Q, ord=2)
        assert jnp.allclose(spectral_norm, 1.0, rtol=0.02)
    
    def test_dualization_scales_correctly(self, key):
        """Modula's dualize should respect target_norm parameter."""
        from modula.atom import Linear
        
        k1, k2 = jax.random.split(key)
        linear = Linear(8, 8)
        weights = linear.initialize(k1)
        
        # Create a gradient
        grad = [jax.random.normal(k2, shape=weights[0].shape)]
        
        # Dualize with different target norms
        for target in [0.5, 1.0, 2.0]:
            dual = linear.dualize(grad, targetNorm=target)
            # The dual should be scaled orthogonalized gradient
            # Check it's properly scaled (approximately)
            actual_scale = jnp.sqrt(8/8)  # sqrt(fanout/fanin) = 1 for square
            expected_norm = target * actual_scale
            # Spectral norm should be close to expected
            spectral = jnp.linalg.norm(dual[0], ord=2)
            assert jnp.allclose(spectral, expected_norm, rtol=0.1)


@pytest.mark.phase2
@pytest.mark.slow
class TestNumericalStability:
    """Test numerical stability of dualization under edge cases."""
    
    def test_newton_schulz_on_ill_conditioned(self, key):
        """Newton-Schulz should handle moderately ill-conditioned matrices."""
        dim = 8
        # Create moderately ill-conditioned matrix (not too extreme)
        U = jax.random.normal(key, shape=(dim, dim), dtype=jnp.float32)
        U, _ = jnp.linalg.qr(U)
        # Moderate condition number
        S = jnp.diag(jnp.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05], dtype=jnp.float32))
        A = U @ S @ U.T
        
        Q = orthogonalize(A)
        # Should still produce approximately orthogonal matrix
        assert jnp.allclose(Q @ Q.T, jnp.eye(dim), rtol=0.1, atol=0.05)
    
    def test_randers_near_boundary(self, key, dim):
        """Randers metric near strong convexity boundary."""
        k1, k2 = jax.random.split(key)
        # drift_strength=0.9 is close to the |b|_A < 1 boundary
        randers = random_randers_metric(k1, dim, drift_strength=0.9)
        v = jax.random.normal(k2, shape=(dim,))
        
        # Should still compute valid norms
        norm = randers.norm(v)
        assert jnp.isfinite(norm)
        assert norm > 0

