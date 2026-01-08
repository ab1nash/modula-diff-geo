"""
Tests for Fisher Metric Extensions (diagonal approximation, sloppy model analysis)

Tests cover:
- Diagonal Fisher approximation
- Natural gradient via diagonal
- Effective dimension computation
- Condition number
- Sloppy model eigenspectrum analysis
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from diffgeo.information.fisher import FisherMetric


class TestDiagonalFisher:
    """Tests for diagonal Fisher approximation."""
    
    def test_diagonal_shape(self):
        """Diagonal should be 1D array of correct size."""
        n = 5
        F = jnp.eye(n) * jnp.arange(1, n + 1)
        metric = FisherMetric(F)
        
        diag = metric.diagonal()
        assert diag.shape == (n,)
    
    def test_diagonal_values(self):
        """Diagonal should extract correct elements."""
        F = jnp.array([
            [1.0, 0.5, 0.2],
            [0.5, 2.0, 0.3],
            [0.2, 0.3, 3.0]
        ])
        metric = FisherMetric(F)
        
        diag = metric.diagonal()
        np.testing.assert_allclose(diag, jnp.array([1.0, 2.0, 3.0]), atol=1e-5)
    
    def test_diagonal_matches_full_for_diagonal_matrix(self):
        """For diagonal F, diagonal approximation should be exact."""
        diag_values = jnp.array([1.0, 2.0, 3.0, 4.0])
        F = jnp.diag(diag_values)
        metric = FisherMetric(F)
        
        gradient = jnp.array([0.4, 0.3, 0.2, 0.1])
        
        # Full natural gradient
        nat_grad_full = metric.natural_gradient(gradient)
        
        # Diagonal natural gradient
        nat_grad_diag = metric.natural_gradient_diagonal(gradient)
        
        np.testing.assert_allclose(nat_grad_full, nat_grad_diag, atol=1e-6)


class TestNaturalGradientDiagonal:
    """Tests for O(n) diagonal natural gradient."""
    
    def test_natural_gradient_diagonal_shape(self):
        """Output should have same shape as input."""
        n = 5
        F = jnp.eye(n) + 0.1 * jnp.ones((n, n))
        metric = FisherMetric(F)
        
        gradient = jnp.ones(n)
        nat_grad = metric.natural_gradient_diagonal(gradient)
        
        assert nat_grad.shape == gradient.shape
    
    def test_natural_gradient_diagonal_formula(self):
        """Should compute g_i / F_ii."""
        F = jnp.array([
            [2.0, 0.5],
            [0.5, 4.0]
        ])
        metric = FisherMetric(F)
        
        gradient = jnp.array([1.0, 2.0])
        nat_grad = metric.natural_gradient_diagonal(gradient)
        
        expected = jnp.array([1.0 / 2.0, 2.0 / 4.0])
        np.testing.assert_allclose(nat_grad, expected, atol=1e-6)
    
    def test_natural_gradient_diagonal_vs_full(self):
        """Diagonal should be an approximation of full natural gradient."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        
        # Create SPD Fisher matrix
        M = jax.random.normal(key1, shape=(4, 4))
        F = M @ M.T + 0.1 * jnp.eye(4)
        metric = FisherMetric(F)
        
        gradient = jax.random.normal(key2, shape=(4,))
        
        nat_grad_full = metric.natural_gradient(gradient)
        nat_grad_diag = metric.natural_gradient_diagonal(gradient)
        
        # Should be correlated but not identical
        correlation = jnp.dot(nat_grad_full, nat_grad_diag) / (
            jnp.linalg.norm(nat_grad_full) * jnp.linalg.norm(nat_grad_diag)
        )
        assert correlation > 0.3, "Diagonal should somewhat approximate full natural gradient"
    
    def test_diagonal_handles_small_values(self):
        """Should handle near-zero diagonal entries gracefully."""
        F = jnp.array([
            [1e-10, 0.0],
            [0.0, 1.0]
        ])
        metric = FisherMetric(F)
        
        gradient = jnp.array([1.0, 1.0])
        
        # Should not produce NaN or Inf
        nat_grad = metric.natural_gradient_diagonal(gradient)
        assert jnp.all(jnp.isfinite(nat_grad))


class TestEffectiveDimension:
    """Tests for sloppy model effective dimension."""
    
    def test_effective_dimension_full_rank(self):
        """Full rank matrix should have eff_dim = n (at low threshold)."""
        n = 5
        F = jnp.eye(n)  # All eigenvalues = 1
        metric = FisherMetric(F)
        
        eff_dim = metric.effective_dimension(threshold=0.1)
        assert eff_dim == n
    
    def test_effective_dimension_low_rank(self):
        """Low rank matrix should have reduced effective dimension."""
        # Rank 2 matrix in 5D
        v1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
        F = jnp.outer(v1, v1) + jnp.outer(v2, v2)
        F = F + 1e-10 * jnp.eye(5)  # Regularize for numerical stability
        
        metric = FisherMetric(F)
        
        eff_dim = metric.effective_dimension(threshold=0.1)
        assert eff_dim == 2
    
    def test_effective_dimension_range(self):
        """Effective dimension should be in [0, n]."""
        key = jax.random.PRNGKey(123)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            M = jax.random.normal(subkey, shape=(5, 5))
            F = M @ M.T + 0.01 * jnp.eye(5)
            metric = FisherMetric(F)
            
            eff_dim = metric.effective_dimension()
            assert 0 <= eff_dim <= 5
    
    def test_effective_dimension_threshold_effect(self):
        """Higher threshold should give lower effective dimension."""
        # Create sloppy Fisher: eigenvalues spanning orders of magnitude
        eigvals = jnp.array([1.0, 0.1, 0.01, 0.001, 0.0001])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        dim_low = metric.effective_dimension(threshold=0.001)
        dim_high = metric.effective_dimension(threshold=0.1)
        
        assert dim_high < dim_low, "Higher threshold should give lower eff dim"


class TestConditionNumber:
    """Tests for condition number computation."""
    
    def test_condition_number_identity(self):
        """Identity matrix should have condition number 1."""
        F = jnp.eye(5)
        metric = FisherMetric(F)
        
        kappa = metric.condition_number()
        np.testing.assert_allclose(kappa, 1.0, atol=1e-5)
    
    def test_condition_number_positive(self):
        """Condition number should be >= 1."""
        key = jax.random.PRNGKey(456)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            M = jax.random.normal(subkey, shape=(4, 4))
            F = M @ M.T + 0.01 * jnp.eye(4)
            metric = FisherMetric(F)
            
            kappa = metric.condition_number()
            assert kappa >= 1.0
    
    def test_condition_number_sloppy_model(self):
        """Sloppy model should have high condition number."""
        # Eigenvalues spanning 6 orders of magnitude
        eigvals = jnp.array([1.0, 1e-2, 1e-4, 1e-6])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        kappa = metric.condition_number()
        assert kappa > 1e5, "Sloppy model should have κ > 10^5"
    
    def test_condition_number_well_conditioned(self):
        """Well-conditioned matrix should have low condition number."""
        eigvals = jnp.array([1.0, 0.9, 0.8, 0.7])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        kappa = metric.condition_number()
        assert kappa < 2, "Well-conditioned matrix should have κ < 2"


class TestEigenspectrum:
    """Tests for eigenspectrum analysis."""
    
    def test_eigenspectrum_sorted(self):
        """Eigenspectrum should be sorted descending."""
        eigvals = jnp.array([0.1, 1.0, 0.5, 0.01])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        spectrum = metric.eigenspectrum()
        
        # Should be descending
        for i in range(len(spectrum) - 1):
            assert spectrum[i] >= spectrum[i + 1]
    
    def test_eigenspectrum_values(self):
        """Eigenspectrum should contain correct eigenvalues."""
        eigvals = jnp.array([3.0, 2.0, 1.0])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        spectrum = metric.eigenspectrum()
        np.testing.assert_allclose(spectrum, jnp.array([3.0, 2.0, 1.0]), atol=1e-6)
    
    def test_eigenspectrum_length(self):
        """Eigenspectrum should have n entries."""
        n = 7
        F = jnp.eye(n)
        metric = FisherMetric(F)
        
        spectrum = metric.eigenspectrum()
        assert len(spectrum) == n


class TestStiffDirections:
    """Tests for stiff direction extraction."""
    
    def test_stiff_directions_shape(self):
        """Stiff directions should have correct shape."""
        F = jnp.eye(5)
        metric = FisherMetric(F)
        
        stiff = metric.stiff_directions(n_directions=3)
        assert stiff.shape == (3, 5)
    
    def test_stiff_directions_normalized(self):
        """Stiff directions should be unit vectors."""
        key = jax.random.PRNGKey(789)
        M = jax.random.normal(key, shape=(4, 4))
        F = M @ M.T + 0.01 * jnp.eye(4)
        metric = FisherMetric(F)
        
        stiff = metric.stiff_directions(n_directions=2)
        
        for i in range(2):
            np.testing.assert_allclose(jnp.linalg.norm(stiff[i]), 1.0, atol=1e-5)
    
    def test_stiff_directions_orthogonal(self):
        """Stiff directions should be orthogonal."""
        key = jax.random.PRNGKey(101)
        M = jax.random.normal(key, shape=(4, 4))
        F = M @ M.T + 0.01 * jnp.eye(4)
        metric = FisherMetric(F)
        
        stiff = metric.stiff_directions(n_directions=3)
        
        # Check pairwise orthogonality
        for i in range(3):
            for j in range(i + 1, 3):
                dot = jnp.dot(stiff[i], stiff[j])
                np.testing.assert_allclose(dot, 0.0, atol=1e-5)
    
    def test_stiff_directions_correspond_to_large_eigenvalues(self):
        """Stiff directions should correspond to largest eigenvalues."""
        # Diagonal Fisher with known eigenvectors
        eigvals = jnp.array([10.0, 5.0, 1.0, 0.1])
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        stiff = metric.stiff_directions(n_directions=2)
        
        # First stiff direction should be [1,0,0,0]
        assert jnp.abs(stiff[0, 0]) > 0.9  # Approximately e1
        # Second should be [0,1,0,0]
        assert jnp.abs(stiff[1, 1]) > 0.9  # Approximately e2


class TestFisherMetricIntegration:
    """Integration tests combining multiple features."""
    
    def test_sloppy_model_diagnostics(self):
        """Full diagnostic workflow for sloppy model."""
        # Create sloppy Fisher (like in systems biology)
        eigvals = 10.0 ** (-jnp.arange(6))  # 10^0, 10^-1, ..., 10^-5
        F = jnp.diag(eigvals)
        metric = FisherMetric(F)
        
        # Condition number should be ~10^5
        kappa = metric.condition_number()
        assert kappa > 1e4
        
        # Effective dimension at 1% threshold should identify few stiff directions
        eff_dim = metric.effective_dimension(threshold=0.01)
        assert eff_dim <= 4  # Most directions are sloppy
        
        # Diagonal natural gradient should work
        gradient = jnp.ones(6)
        nat_grad = metric.natural_gradient_diagonal(gradient)
        assert jnp.all(jnp.isfinite(nat_grad))
    
    def test_diagonal_vs_full_optimization_direction(self):
        """Diagonal and full should give similar optimization directions."""
        key = jax.random.PRNGKey(999)
        key1, key2 = jax.random.split(key)
        
        # Random SPD Fisher
        M = jax.random.normal(key1, shape=(5, 5))
        F = M @ M.T + 0.5 * jnp.eye(5)
        metric = FisherMetric(F)
        
        gradient = jax.random.normal(key2, shape=(5,))
        
        ng_full = metric.natural_gradient(gradient)
        ng_diag = metric.natural_gradient_diagonal(gradient)
        
        # Should point in similar directions
        cos_angle = jnp.dot(ng_full, ng_diag) / (
            jnp.linalg.norm(ng_full) * jnp.linalg.norm(ng_diag)
        )
        assert cos_angle > 0.3, "Should be correlated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

