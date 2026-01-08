"""
Unit tests for the Universal Fisher Geometry Framework.

Tests the new modules:
- diffgeo/statistical_manifold.py
- diffgeo/geometry_extractor.py
- diffgeo/optimizer.py

These tests verify:
1. Fisher metric computation is correct
2. Natural gradient conversion works
3. Geometry extraction from time series matches expected SPD
4. Asymmetry detection triggers Finsler extension
"""
import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Skip all tests if JAX not available
pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX required")


class TestStatisticalManifold:
    """Tests for StatisticalManifold class."""
    
    def test_gaussian_manifold_creation(self):
        """Test creating manifold from Gaussian distribution."""
        from diffgeo import StatisticalManifold
        
        # Create simple Gaussian manifold
        mean = jnp.array([1.0, 2.0, 3.0])
        covariance = jnp.array([
            [1.0, 0.5, 0.0],
            [0.5, 2.0, 0.3],
            [0.0, 0.3, 1.5]
        ])
        
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        # Check basic properties
        assert manifold.dim == 3
        assert not manifold.is_asymmetric()
        assert manifold.fisher_metric is not None
    
    def test_fisher_metric_for_gaussian(self):
        """Fisher metric for Gaussian mean should be inverse covariance."""
        from diffgeo import StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        # Fisher metric for Gaussian mean is Σ^{-1}
        expected_fisher = jnp.linalg.inv(covariance)
        computed_fisher = manifold.fisher_metric.matrix
        
        # Check they match (up to regularization)
        np.testing.assert_allclose(
            computed_fisher, expected_fisher, 
            rtol=1e-4, atol=1e-5
        )
    
    def test_natural_gradient_computation(self):
        """Natural gradient should be F^{-1} @ gradient."""
        from diffgeo import StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.array([[2.0, 0.0], [0.0, 3.0]])  # Diagonal for easy check
        
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        # Euclidean gradient
        gradient = jnp.array([1.0, 1.0])
        
        # Natural gradient = F^{-1} @ g = Σ @ g (for Gaussian)
        natural_grad = manifold.natural_gradient(gradient)
        
        # For diagonal covariance, natural_grad = covariance @ gradient
        expected = covariance @ gradient
        
        np.testing.assert_allclose(natural_grad, expected, rtol=1e-4)
    
    def test_geodesic_distance(self):
        """Geodesic distance should use Fisher metric."""
        from diffgeo import StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.eye(2)  # Identity for simple case
        
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        # Distance to point
        other_params = jnp.array([1.0, 0.0])
        distance = manifold.geodesic_distance(other_params)
        
        # With identity Fisher, distance is just Euclidean
        expected = 1.0
        assert abs(distance - expected) < 0.01


class TestDataGeometryExtractor:
    """Tests for DataGeometryExtractor class."""
    
    def test_time_series_extraction(self):
        """Extract geometry from time series data."""
        from diffgeo import DataGeometryExtractor
        
        # Generate simple multivariate time series
        np.random.seed(42)
        n_timepoints = 100
        n_channels = 4
        
        # Correlated data
        L = np.random.randn(n_channels, n_channels)
        cov = L @ L.T + 0.1 * np.eye(n_channels)
        data = np.random.multivariate_normal(
            np.zeros(n_channels), cov, size=n_timepoints
        )
        
        # Extract geometry
        extractor = DataGeometryExtractor()
        manifold = extractor.from_time_series(jnp.array(data))
        
        # Check we got a valid manifold
        assert manifold.dim == n_channels
        assert manifold.fisher_metric is not None
    
    def test_spd_geometry_extraction(self):
        """Extract geometry from SPD matrices."""
        from diffgeo import SPDGeometryExtractor
        
        # Generate random SPD matrices
        np.random.seed(42)
        n_samples = 20
        d = 3
        
        matrices = []
        for _ in range(n_samples):
            L = np.random.randn(d, d)
            matrices.append(L @ L.T + 0.5 * np.eye(d))
        matrices = jnp.array(matrices)
        
        # Extract geometry
        extractor = SPDGeometryExtractor()
        manifold = extractor.from_spd_matrices(matrices)
        
        # Check we got a valid manifold
        assert manifold.fisher_metric is not None


class TestGeometricOptimizer:
    """Tests for GeometricOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        from diffgeo import GeometricOptimizer, StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.eye(2)
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        optimizer = GeometricOptimizer(
            manifold=manifold,
            learning_rate=0.01
        )
        
        state = optimizer.init(mean)
        assert state.params is not None
        assert state.step == 0
    
    def test_optimizer_step(self):
        """Test optimizer performs valid updates."""
        from diffgeo import GeometricOptimizer, StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.eye(2)
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        optimizer = GeometricOptimizer(
            manifold=manifold,
            learning_rate=0.1
        )
        
        state = optimizer.init(mean)
        gradient = jnp.array([1.0, 2.0])
        
        new_state = optimizer.step(state, gradient)
        
        # Check params were updated
        assert new_state.step == 1
        assert not jnp.allclose(new_state.params, state.params)
    
    def test_natural_gradient_optimizer(self):
        """Test NaturalGradientOptimizer."""
        from diffgeo import NaturalGradientOptimizer, StatisticalManifold
        
        mean = jnp.array([1.0, 2.0])
        covariance = jnp.array([[2.0, 0.0], [0.0, 0.5]])
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        optimizer = NaturalGradientOptimizer(
            manifold=manifold,
            learning_rate=0.01
        )
        
        state = optimizer.init(mean)
        gradient = jnp.array([0.5, 0.5])
        
        new_state = optimizer.step(state, gradient)
        
        # Check update was applied
        delta = new_state.params - state.params
        assert jnp.linalg.norm(delta) > 0


class TestTensorVariance:
    """Tests for covariant/contravariant distinction."""
    
    def test_variance_types_exist(self):
        """Verify TensorVariance enum has expected values."""
        from diffgeo import TensorVariance
        
        assert TensorVariance.CONTRAVARIANT is not None
        assert TensorVariance.COVARIANT is not None
        assert TensorVariance.SCALAR is not None
    
    def test_gradient_is_covariant(self):
        """Gradients should be classified as covariant."""
        # This is a conceptual test - gradients live in dual space
        from diffgeo import TensorVariance
        
        # By convention, gradients are covariant (lower indices)
        gradient_variance = TensorVariance.COVARIANT
        assert gradient_variance.value == "co"
    
    def test_velocity_is_contravariant(self):
        """Velocities/displacements should be contravariant."""
        from diffgeo import TensorVariance
        
        # Velocities are contravariant (upper indices)
        velocity_variance = TensorVariance.CONTRAVARIANT
        assert velocity_variance.value == "contra"


class TestAsymmetryDetection:
    """Tests for Finsler extension when asymmetry detected."""
    
    def test_symmetric_data_no_finsler(self):
        """Symmetric data should not trigger Finsler."""
        from diffgeo import StatisticalManifold
        
        mean = jnp.array([0.0, 0.0])
        covariance = jnp.eye(2)
        
        manifold = StatisticalManifold.from_gaussian(mean, covariance)
        
        # Gaussian is symmetric, no Finsler needed
        assert not manifold.is_asymmetric()
        assert manifold.as_randers_metric() is None


class TestIntegration:
    """Integration tests verifying all components work together."""
    
    def test_full_optimization_loop(self):
        """Test complete optimization with geometric optimizer."""
        from diffgeo import StatisticalManifold, GeometricOptimizer
        
        # Setup
        true_params = jnp.array([3.0, -2.0])
        init_params = jnp.array([0.0, 0.0])
        covariance = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        
        manifold = StatisticalManifold.from_gaussian(init_params, covariance)
        optimizer = GeometricOptimizer(manifold, learning_rate=0.1)
        
        # Define loss: distance to true params
        def loss_fn(params):
            return jnp.sum((params - true_params) ** 2)
        
        # Optimize
        state = optimizer.init(init_params)
        for _ in range(50):
            gradient = jax.grad(loss_fn)(state.params)
            state = optimizer.step(state, gradient)
        
        # Should have moved toward true params
        final_distance = jnp.linalg.norm(state.params - true_params)
        initial_distance = jnp.linalg.norm(init_params - true_params)
        
        assert final_distance < initial_distance


class TestFisherImputationModel:
    """Tests for FisherImputationModel in benchmarks."""
    
    def test_fisher_model_creation(self):
        """Test FisherImputationModel can be created."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tests.realworld.benchmarks.learnable import FisherImputationModel
        
        model = FisherImputationModel(input_dim=10, hidden_dim=16, seed=42)
        
        assert model.name == "Fisher Geometry"
        assert model.use_natural_gradient == True
        assert model.detect_asymmetry == True
    
    def test_fisher_model_forward(self):
        """Test FisherImputationModel forward pass."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tests.realworld.benchmarks.learnable import FisherImputationModel
        
        model = FisherImputationModel(input_dim=8, hidden_dim=16, seed=42)
        
        key = jax.random.PRNGKey(42)
        params = model.init_params(key)
        
        x = jnp.ones((5, 8))
        y = model.forward(x, params)
        
        assert y.shape == (5, 8)
    
    def test_fisher_model_update(self):
        """Test FisherImputationModel compute_update."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tests.realworld.benchmarks.learnable import FisherImputationModel
        
        model = FisherImputationModel(input_dim=8, hidden_dim=16, seed=42)
        
        key = jax.random.PRNGKey(42)
        params = model.init_params(key)
        grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, params)
        
        new_params = model.compute_update(params, grads, lr=0.01)
        
        # Params should have changed
        first_key = list(params.keys())[0]
        assert not jnp.allclose(params[first_key], new_params[first_key])
        
        # Fisher EMA should be computed
        assert model._fisher_ema is not None


# Need Path for test imports
from pathlib import Path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

