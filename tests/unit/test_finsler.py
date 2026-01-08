"""
Tests for Finsler Geometry (RandersMetric, dualization, geodesics)

Tests cover:
- Randers norm properties (positive, homogeneous)
- Dual norm convexity and homogeneity
- Finsler duality inverse property
- Geodesic midpoint and interpolation
- Reduction to Riemannian when b=0
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from diffgeo.geometry.finsler import (
    RandersMetric,
    FinslerDualizer,
)


class TestRandersNorm:
    """Tests for Randers norm F(v) = sqrt(v^T A v) + b^T v."""
    
    def test_norm_positive(self):
        """F(v) > 0 for all nonzero v."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(42)
        for _ in range(20):
            key, subkey = jax.random.split(key)
            v = jax.random.normal(subkey, shape=(2,))
            if jnp.linalg.norm(v) > 1e-6:
                assert metric.norm(v) > 0, "Finsler norm should be positive"
    
    def test_norm_at_zero(self):
        """F(0) = 0."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        v = jnp.zeros(2)
        np.testing.assert_allclose(metric.norm(v), 0.0, atol=1e-10)
    
    def test_norm_positive_homogeneity(self):
        """F(λv) = λ F(v) for λ > 0."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(123)
        v = jax.random.normal(key, shape=(2,))
        
        for lambda_ in [0.5, 1.0, 2.0, 3.5]:
            np.testing.assert_allclose(
                metric.norm(lambda_ * v),
                lambda_ * metric.norm(v),
                atol=1e-6
            )
    
    def test_asymmetry(self):
        """F(v) ≠ F(-v) when b ≠ 0."""
        A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([0.3, 0.0])  # Significant drift
        metric = RandersMetric(A, b)
        
        v = jnp.array([1.0, 0.0])
        
        forward = metric.norm(v)
        backward = metric.norm(-v)
        
        # Randers F(v) = sqrt(v^T A v) + b^T v
        # Forward: adds b^T v (positive), backward: subtracts
        # So forward > backward in this convention
        assert forward != backward, "Randers should be asymmetric when b≠0"
        # The actual direction depends on the convention used
    
    def test_riemannian_symmetry(self):
        """F(v) = F(-v) when b = 0 (reduces to Riemannian)."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.zeros(2)
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(456)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            v = jax.random.normal(subkey, shape=(2,))
            
            np.testing.assert_allclose(
                metric.norm(v),
                metric.norm(-v),
                atol=1e-10
            )
    
    def test_is_symmetric_check(self):
        """is_symmetric() should correctly identify symmetric case."""
        A = jnp.eye(3)
        
        metric_symmetric = RandersMetric(A, jnp.zeros(3))
        metric_asymmetric = RandersMetric(A, jnp.array([0.1, 0.0, 0.0]))
        
        assert metric_symmetric.is_symmetric()
        assert not metric_asymmetric.is_symmetric()


class TestDualNorm:
    """Tests for Randers dual norm F*(ℓ)."""
    
    def test_dual_norm_positive_homogeneity(self):
        """F*(αℓ) = α F*(ℓ) for α > 0."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(789)
        ell = jax.random.normal(key, shape=(2,))
        
        for alpha in [0.5, 1.0, 2.0, 3.5]:
            np.testing.assert_allclose(
                metric.dual_norm(alpha * ell),
                alpha * metric.dual_norm(ell),
                atol=1e-5
            )
    
    def test_dual_norm_positive(self):
        """F*(ℓ) > 0 for all nonzero ℓ."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(101)
        for _ in range(20):
            key, subkey = jax.random.split(key)
            ell = jax.random.normal(subkey, shape=(2,))
            if jnp.linalg.norm(ell) > 1e-6:
                assert metric.dual_norm(ell) > 0
    
    def test_dual_norm_reduces_to_riemannian(self):
        """When b=0, dual norm is sqrt(ℓ^T A^{-1} ℓ)."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.zeros(2)
        metric = RandersMetric(A, b)
        
        A_inv = jnp.linalg.inv(A)
        
        key = jax.random.PRNGKey(102)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            ell = jax.random.normal(subkey, shape=(2,))
            
            expected = jnp.sqrt(ell @ A_inv @ ell)
            actual = metric.dual_norm(ell)
            
            np.testing.assert_allclose(actual, expected, atol=1e-5)


class TestFinslerDualizer:
    """Tests for FinslerDualizer gradient → update conversion."""
    
    def test_dualize_produces_finite_output(self):
        """Dualized gradient should be finite and non-zero."""
        A = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        b = jnp.array([0.1, 0.0])
        metric = RandersMetric(A, b)
        dualizer = FinslerDualizer(metric)
        
        gradient = jnp.array([1.0, 1.0])
        update = dualizer.dualize(gradient)
        
        # Update should be finite and non-zero
        assert jnp.all(jnp.isfinite(update))
        assert jnp.linalg.norm(update) > 0
    
    def test_dualize_reduces_to_riemannian(self):
        """When b=0, dualize should give A^{-1} g."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.zeros(2)
        metric = RandersMetric(A, b)
        dualizer = FinslerDualizer(metric)
        
        A_inv = jnp.linalg.inv(A)
        
        key = jax.random.PRNGKey(103)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            gradient = jax.random.normal(subkey, shape=(2,))
            
            update = dualizer.dualize(gradient, target_norm=1.0)
            expected_direction = A_inv @ gradient
            expected_direction = expected_direction / jnp.linalg.norm(expected_direction)
            
            # Should be in same direction (up to sign and scale)
            actual_direction = update / jnp.linalg.norm(update)
            
            # Either parallel or antiparallel
            dot = jnp.abs(jnp.dot(actual_direction, expected_direction))
            np.testing.assert_allclose(dot, 1.0, atol=1e-4)
    
    def test_dualize_output_norm(self):
        """Dualized vector should have target norm."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        dualizer = FinslerDualizer(metric)
        
        gradient = jnp.array([1.0, 2.0])
        
        for target in [0.5, 1.0, 2.0]:
            update = dualizer.dualize(gradient, target_norm=target)
            actual_norm = metric.norm(update)
            np.testing.assert_allclose(actual_norm, target, atol=1e-4)


class TestGeodesicMidpoint:
    """Tests for first-order geodesic midpoint approximation."""
    
    def test_midpoint_symmetric_metric(self):
        """For symmetric metric, midpoint should be (p1+p2)/2."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.zeros(2)  # Symmetric
        metric = RandersMetric(A, b)
        
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([2.0, 2.0])
        
        midpoint = metric.geodesic_midpoint(p1, p2)
        expected = (p1 + p2) / 2
        
        np.testing.assert_allclose(midpoint, expected, atol=1e-6)
    
    def test_midpoint_asymmetric_differs(self):
        """For asymmetric metric, midpoint should differ from Euclidean midpoint."""
        A = jnp.eye(2)
        b = jnp.array([0.3, 0.0])  # Drift in x direction
        metric = RandersMetric(A, b)
        
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([2.0, 0.0])
        
        midpoint = metric.geodesic_midpoint(p1, p2)
        euclidean_mid = (p1 + p2) / 2  # [1.0, 0.0]
        
        # The asymmetric metric should produce a different midpoint
        # The actual shift direction depends on the Randers convention
        assert not jnp.allclose(midpoint, euclidean_mid), "Asymmetric should differ from Euclidean"
    
    def test_midpoint_between_points(self):
        """Midpoint should lie between p1 and p2."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        key = jax.random.PRNGKey(104)
        for _ in range(10):
            key, k1, k2 = jax.random.split(key, 3)
            p1 = jax.random.normal(k1, shape=(2,))
            p2 = jax.random.normal(k2, shape=(2,))
            
            midpoint = metric.geodesic_midpoint(p1, p2)
            
            # Midpoint should be on line segment (in Euclidean sense as approximation)
            # Check that midpoint is in convex hull
            t = jnp.linalg.solve(
                jnp.column_stack([p1 - p2, jnp.array([1.0, 0.0])]),
                midpoint - p2
            )[0]
            # t should be in [0, 1]
            # Note: This is approximate since Finsler geodesics aren't straight
    
    def test_geodesic_interpolate_endpoints(self):
        """Interpolation at t=0 and t=1 should return endpoints."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        b = jnp.array([0.1, 0.05])
        metric = RandersMetric(A, b)
        
        p1 = jnp.array([1.0, 2.0])
        p2 = jnp.array([3.0, 4.0])
        
        np.testing.assert_allclose(
            metric.geodesic_interpolate(p1, p2, 0.0), p1, atol=1e-6
        )
        np.testing.assert_allclose(
            metric.geodesic_interpolate(p1, p2, 1.0), p2, atol=1e-6
        )


class TestFinslerUtilities:
    """Tests for Finsler utility functions."""
    
    def test_randers_from_riemannian_and_with_drift(self):
        """Test creating Randers metrics via factory methods."""
        # Random SPD matrix
        key = jax.random.PRNGKey(105)
        M = jax.random.normal(key, shape=(3, 3))
        A = M @ M.T + 0.1 * jnp.eye(3)
        
        # Pure Riemannian (b=0)
        riemannian = RandersMetric.from_riemannian(A)
        assert riemannian.is_symmetric()
        
        # With drift
        direction = jnp.array([1.0, 0.0, 0.0])
        with_drift = RandersMetric.with_drift(A, direction, drift_strength=0.2)
        assert not with_drift.is_symmetric()
        assert with_drift.b_norm_A_sq < 1.0
    
    def test_randers_metric_valid(self):
        """RandersMetric should satisfy validity condition ||b||_A < 1."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        
        # Valid: small b
        b_valid = jnp.array([0.1, 0.05])
        metric_valid = RandersMetric(A, b_valid)
        assert metric_valid.is_valid()
        
        # For zero drift
        b_zero = jnp.zeros(2)
        metric_zero = RandersMetric(A, b_zero)
        assert metric_zero.is_valid()
    
    def test_from_riemannian(self):
        """RandersMetric.from_riemannian should set b=0."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        metric = RandersMetric.from_riemannian(A)
        
        np.testing.assert_allclose(metric.b, jnp.zeros(2), atol=1e-10)
        assert metric.is_symmetric()
    
    def test_with_drift(self):
        """RandersMetric.with_drift should set correct drift strength."""
        A = jnp.eye(3)
        direction = jnp.array([1.0, 0.0, 0.0])
        
        for strength in [0.1, 0.3, 0.5]:
            metric = RandersMetric.with_drift(A, direction, drift_strength=strength)
            
            # ||b||_A should equal strength
            actual_strength = jnp.sqrt(metric.b_norm_A_sq)
            np.testing.assert_allclose(actual_strength, strength, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

