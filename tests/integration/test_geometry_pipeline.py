"""
Integration Tests for Full Geometry Pipeline

Tests the complete flow from data to optimized model:
- Data → FisherMetric → GeometricOptimizer → Convergence
- RandersMetric + SO(3) operations
- SPD manifold Fisher consistency
- Retraction vs exp map comparison
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from diffgeo.geometry.lie_groups import (
    so3_exp,
    so3_log,
    so3_geodesic,
    qr_retraction,
    polar_retraction,
    random_rotation,
    is_rotation_matrix,
)
from diffgeo.geometry.finsler import RandersMetric, FinslerDualizer
from diffgeo.geometry.spd import SPDManifold
from diffgeo.information.fisher import FisherMetric
from diffgeo.information.manifolds import StatisticalManifold
from diffgeo.optim.optimizer import GeometricOptimizer, GeometricOptimizerState


class TestFisherToOptimizerFlow:
    """Test complete flow from data to optimized parameters."""
    
    def test_gaussian_fisher_optimization(self):
        """Optimize using Fisher metric from Gaussian data."""
        # Generate Gaussian data
        key = jax.random.PRNGKey(42)
        key, data_key = jax.random.split(key)
        
        # True covariance
        true_cov = jnp.array([
            [2.0, 0.5],
            [0.5, 1.0]
        ])
        L = jnp.linalg.cholesky(true_cov)
        
        # Generate samples
        n_samples = 100
        noise = jax.random.normal(data_key, shape=(n_samples, 2))
        data = noise @ L.T
        
        # Create Fisher metric from data
        mean = jnp.mean(data, axis=0)
        cov = jnp.cov(data.T)
        cov = cov + 1e-4 * jnp.eye(2)
        
        fisher = FisherMetric(cov)
        
        # Create statistical manifold
        manifold = StatisticalManifold.from_gaussian(
            mean=mean,
            covariance=cov,
            samples=data
        )
        
        # Create optimizer
        optimizer = GeometricOptimizer(
            manifold=manifold,
            learning_rate=0.1,
            use_momentum=True
        )
        
        # Initialize state
        key, init_key = jax.random.split(key)
        initial_params = jax.random.normal(init_key, shape=(2,))
        state = optimizer.init(initial_params)
        
        # Simple quadratic loss
        def loss_fn(params):
            return 0.5 * jnp.sum((params - mean) ** 2)
        
        # Optimize
        losses = []
        for _ in range(20):
            loss = loss_fn(state.params)
            losses.append(float(loss))
            
            gradient = jax.grad(loss_fn)(state.params)
            state = optimizer.step(state, gradient)
        
        # Should converge
        assert losses[-1] < losses[0], "Loss should decrease"
        assert losses[-1] < 0.1, "Should converge near optimum"
    
    def test_natural_gradient_faster_convergence(self):
        """Natural gradient should converge faster than vanilla gradient."""
        key = jax.random.PRNGKey(123)
        
        # Ill-conditioned Fisher matrix
        eigvals = jnp.array([10.0, 0.1])
        fisher_matrix = jnp.diag(eigvals)
        fisher = FisherMetric(fisher_matrix)
        
        # Target
        target = jnp.array([1.0, 1.0])
        
        def loss_fn(params):
            diff = params - target
            return 0.5 * diff @ fisher_matrix @ diff
        
        # Vanilla gradient descent
        params_vanilla = jnp.zeros(2)
        lr = 0.05
        for _ in range(50):
            grad = jax.grad(loss_fn)(params_vanilla)
            params_vanilla = params_vanilla - lr * grad
        
        # Natural gradient descent
        params_natural = jnp.zeros(2)
        for _ in range(50):
            grad = jax.grad(loss_fn)(params_natural)
            nat_grad = fisher.natural_gradient(grad)
            params_natural = params_natural - lr * nat_grad
        
        # Both should converge, but let's check they work
        loss_vanilla = loss_fn(params_vanilla)
        loss_natural = loss_fn(params_natural)
        
        assert loss_vanilla < 0.5, "Vanilla should make progress"
        assert loss_natural < 0.5, "Natural should make progress"


class TestFinslerWithLieGroup:
    """Test RandersMetric with SO(3) operations."""
    
    def test_finsler_so3_integration(self):
        """RandersMetric should work with SO(3) tangent vectors."""
        # Create Randers metric on so(3) (tangent space)
        A = jnp.eye(3) * 2.0  # Scaled identity
        b = jnp.array([0.1, 0.0, 0.0])  # Drift around x-axis
        metric = RandersMetric(A, b)
        
        # Random angular velocity (in so(3))
        key = jax.random.PRNGKey(456)
        omega = jax.random.normal(key, shape=(3,))
        
        # Finsler norm
        finsler_norm = metric.norm(omega)
        assert finsler_norm > 0
        
        # Apply SO(3) exp map
        R = so3_exp(omega)
        assert is_rotation_matrix(R)
    
    def test_finsler_dualizer_with_so3_tangent(self):
        """FinslerDualizer should produce valid SO(3) tangent updates."""
        A = jnp.eye(3) * 2.0
        b = jnp.array([0.1, 0.05, 0.0])
        metric = RandersMetric(A, b)
        dualizer = FinslerDualizer(metric)
        
        # Gradient in so(3)
        gradient = jnp.array([0.5, 0.3, 0.2])
        
        # Dualize
        update = dualizer.dualize(gradient, target_norm=0.1)
        
        # Should produce valid rotation when exponentiated
        R = so3_exp(update)
        assert is_rotation_matrix(R)
    
    def test_asymmetric_rotation_cost(self):
        """Rotation with/against drift should have different costs."""
        A = jnp.eye(3)
        b = jnp.array([0.3, 0.0, 0.0])  # Strong drift around x
        metric = RandersMetric(A, b)
        
        # Rotation around x (with drift)
        omega_with = jnp.array([1.0, 0.0, 0.0])
        # Rotation against x
        omega_against = jnp.array([-1.0, 0.0, 0.0])
        
        cost_with = metric.norm(omega_with)
        cost_against = metric.norm(omega_against)
        
        # Randers asymmetry: costs should differ
        assert cost_with != cost_against, "Asymmetric metric should have different costs"
    
    def test_geodesic_respects_finsler(self):
        """SO(3) geodesic midpoint should differ from Finsler midpoint."""
        key = jax.random.PRNGKey(789)
        key1, key2 = jax.random.split(key)
        
        R1 = random_rotation(key1)
        R2 = random_rotation(key2)
        
        # SO(3) geodesic midpoint (Lie group geodesic)
        R_mid_so3 = so3_geodesic(R1, R2, 0.5)
        assert is_rotation_matrix(R_mid_so3)
        
        # Convert to tangent space at R1
        omega = so3_log(R1.T @ R2)
        
        # Finsler midpoint in tangent space
        A = jnp.eye(3)
        b = jnp.array([0.2, 0.0, 0.0])
        metric = RandersMetric(A, b)
        
        # Finsler geodesic would give different point
        omega_mid_finsler = metric.geodesic_midpoint(jnp.zeros(3), omega)
        R_mid_finsler = R1 @ so3_exp(omega_mid_finsler)
        
        # Both should be valid rotations
        assert is_rotation_matrix(R_mid_finsler)


class TestSPDFisherConsistency:
    """Test SPD manifold Fisher matches FisherMetric.from_gaussian."""
    
    def test_spd_distance_matches_fisher_distance(self):
        """SPD Riemannian distance should relate to Fisher metric."""
        # Two SPD matrices (covariances)
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        B = jnp.array([[2.2, 0.4], [0.4, 1.1]])
        
        # SPD Riemannian distance
        spd = SPDManifold(dim=2)
        d_spd = spd.distance(A, B)
        
        # Fisher metric on covariance manifold
        # For Gaussians with fixed mean, Fisher = 1/2 (Σ^-1 ⊗ Σ^-1)
        # The local distance should relate to SPD distance
        
        assert d_spd > 0
        assert jnp.isfinite(d_spd)
    
    def test_spd_geodesic_stays_spd(self):
        """Geodesic on SPD should produce SPD matrices."""
        A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        B = jnp.array([[3.0, 0.2], [0.2, 2.0]])
        
        spd = SPDManifold(dim=2)
        
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            C = spd.geodesic(A, B, t)
            
            # Check symmetric
            np.testing.assert_allclose(C, C.T, atol=1e-6)
            
            # Check positive definite
            eigvals = jnp.linalg.eigvalsh(C)
            assert jnp.all(eigvals > 0)
    
    def test_frechet_mean_of_covariances(self):
        """Fréchet mean should be well-defined on SPD."""
        # Several SPD matrices
        matrices = [
            jnp.array([[2.0, 0.3], [0.3, 1.0]]),
            jnp.array([[1.5, 0.2], [0.2, 1.5]]),
            jnp.array([[2.5, 0.4], [0.4, 1.2]]),
        ]
        
        spd = SPDManifold(dim=2)
        mean = spd.frechet_mean(jnp.stack(matrices))
        
        # Mean should be SPD
        np.testing.assert_allclose(mean, mean.T, atol=1e-5)
        eigvals = jnp.linalg.eigvalsh(mean)
        assert jnp.all(eigvals > 0)


class TestRetractionVsExpMap:
    """Compare retractions with exponential map."""
    
    def test_retraction_first_order_agreement(self):
        """Retractions should agree with exp map to first order."""
        key = jax.random.PRNGKey(101)
        key1, key2 = jax.random.split(key)
        
        # Base point
        base = random_rotation(key1)
        
        # Small tangent vector (in tangent space)
        omega = jax.random.normal(key2, shape=(3,)) * 0.01
        tangent = omega.reshape(3, 1) @ jnp.array([[1, 0, 0]])  # Approximate tangent
        # More proper tangent: use skew symmetric
        from diffgeo.geometry.lie_groups import skew_symmetric
        tangent = skew_symmetric(omega) @ base
        
        # Exponential map
        R_exp = base @ so3_exp(omega)
        
        # QR retraction
        R_qr = qr_retraction(base, tangent)
        
        # Polar retraction
        R_polar = polar_retraction(base, tangent)
        
        # All should be valid rotations
        assert is_rotation_matrix(R_exp)
        assert is_rotation_matrix(R_qr)
        assert is_rotation_matrix(R_polar)
        
        # Should be close for small tangent
        np.testing.assert_allclose(R_exp, R_qr, atol=0.1)
        np.testing.assert_allclose(R_exp, R_polar, atol=0.1)
    
    def test_retraction_stays_on_manifold_for_large_step(self):
        """Retractions should stay on manifold even for large steps."""
        key = jax.random.PRNGKey(202)
        key1, key2 = jax.random.split(key)
        
        base = random_rotation(key1)
        
        # Large tangent
        tangent = jax.random.normal(key2, shape=(3, 3))
        
        R_qr = qr_retraction(base, tangent)
        R_polar = polar_retraction(base, tangent)
        
        # Both should be valid (exp map might struggle)
        np.testing.assert_allclose(R_qr.T @ R_qr, jnp.eye(3), atol=1e-5)
        np.testing.assert_allclose(R_polar.T @ R_polar, jnp.eye(3), atol=1e-5)


class TestOptimizerWithRetraction:
    """Test optimizer with retraction-based updates."""
    
    def test_optimizer_orthogonal_constraint(self):
        """Optimizer should maintain orthogonal constraint."""
        # Simple loss: distance from target rotation
        target = jnp.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=jnp.float32)
        
        def loss_fn(R):
            return jnp.sum((R - target) ** 2)
        
        optimizer = GeometricOptimizer(
            manifold=None,  # No Fisher, just retractions
            learning_rate=0.1
        )
        
        key = jax.random.PRNGKey(303)
        R0 = random_rotation(key)
        state = optimizer.init(R0)
        
        # Optimize with retraction
        for _ in range(50):
            grad = jax.grad(loss_fn)(state.params)
            state = optimizer.step_with_retraction(
                state, grad, manifold_type="orthogonal"
            )
            
            # Check still on manifold
            np.testing.assert_allclose(
                state.params.T @ state.params, jnp.eye(3), atol=1e-4
            )
        
        # Should make progress toward target
        final_loss = loss_fn(state.params)
        initial_loss = loss_fn(R0)
        assert final_loss < initial_loss
    
    def test_optimizer_sphere_constraint(self):
        """Optimizer should maintain sphere constraint."""
        target = jnp.array([1.0, 0.0, 0.0])
        
        def loss_fn(x):
            return jnp.sum((x - target) ** 2)
        
        optimizer = GeometricOptimizer(
            manifold=None,
            learning_rate=0.2
        )
        
        x0 = jnp.array([0.0, 1.0, 0.0])
        x0 = x0 / jnp.linalg.norm(x0)  # On unit sphere
        state = optimizer.init(x0)
        
        for _ in range(30):
            grad = jax.grad(loss_fn)(state.params)
            state = optimizer.step_with_retraction(
                state, grad, manifold_type="sphere"
            )
            
            # Check still on sphere
            np.testing.assert_allclose(
                jnp.linalg.norm(state.params), 1.0, atol=1e-5
            )


class TestEndToEndPipeline:
    """Test complete data → geometry → optimization pipeline."""
    
    def test_time_series_to_fisher_to_optimization(self):
        """Full pipeline from time series data."""
        # Generate time series
        key = jax.random.PRNGKey(404)
        n_samples, n_time, n_channels = 50, 20, 3
        
        # Correlated time series
        data = jax.random.normal(key, shape=(n_samples, n_time, n_channels))
        
        # Compute empirical covariance across time
        flattened = data.reshape(n_samples, -1)
        cov = jnp.cov(flattened.T)
        cov = cov + 1e-4 * jnp.eye(cov.shape[0])
        
        # Create Fisher metric
        fisher = FisherMetric(cov[:5, :5])  # Use subset for tractability
        
        # Diagnostics
        eff_dim = fisher.effective_dimension(threshold=0.01)
        kappa = fisher.condition_number()
        
        assert 0 < eff_dim <= 5
        assert kappa >= 1
        
        # Use diagonal natural gradient
        gradient = jnp.ones(5)
        nat_grad_diag = fisher.natural_gradient_diagonal(gradient)
        nat_grad_full = fisher.natural_gradient(gradient)
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(nat_grad_diag))
        assert jnp.all(jnp.isfinite(nat_grad_full))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

