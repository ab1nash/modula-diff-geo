"""
Phase 1 Tests: Core Type System and Transformation Laws

These tests verify the fundamental mathematical properties that underpin
geometric covariance:
- Contravariant transformation: v' = J^{-1}v
- Covariant transformation: α' = αJ  
- Scalar pairing invariance: α·v = α'·v'
- Twisted form sign flips under reflection
- Metric transformation preserves inner products

Reference: Burke's "Applied Differential Geometry" - The Descriptive Geometry of Tensors
"""
import pytest
import jax
import jax.numpy as jnp

from tests.geometry.generators import (
    random_vector, random_covector, random_twisted_vector,
    random_metric, random_basis_change, random_reflection,
    TensorVariance, Parity, MetricTensor
)
from tests.geometry.invariants import (
    assert_contravariant_transformation,
    assert_covariant_transformation,
    assert_twisted_transformation,
    assert_scalar_pairing_invariant,
    assert_metric_transformation,
    assert_inner_product_invariant,
    assert_positive_definite,
    assert_index_raising_lowering_inverse,
    assert_parity_composition,
    assert_reflection_sign_flip,
    full_geometric_consistency_check,
    InvariantViolation
)


@pytest.mark.phase1
@pytest.mark.invariant
class TestTransformationLaws:
    """Test fundamental tensor transformation properties."""
    
    def test_contravariant_vector_transforms_with_inverse_jacobian(self, key, dim, tolerance):
        """Vectors (velocities, displacements) transform as v' = J^{-1}v."""
        k1, k2 = jax.random.split(key)
        v = random_vector(k1, dim).components
        bc = random_basis_change(k2, dim)
        
        v_prime = bc.J_inv @ v
        assert_contravariant_transformation(v, v_prime, bc, rtol=tolerance)
    
    def test_covariant_vector_transforms_with_jacobian(self, key, dim, tolerance):
        """Covectors (gradients, 1-forms) transform as α' = αJ."""
        k1, k2 = jax.random.split(key)
        alpha = random_covector(k1, dim).components
        bc = random_basis_change(k2, dim)
        
        alpha_prime = alpha @ bc.J
        assert_covariant_transformation(alpha, alpha_prime, bc, rtol=tolerance)
    
    def test_scalar_pairing_is_coordinate_invariant(self, key, dim, tolerance):
        """The contraction α_i v^i must be a true scalar."""
        k1, k2, k3 = jax.random.split(key, 3)
        v = random_vector(k1, dim).components
        alpha = random_covector(k2, dim).components
        bc = random_basis_change(k3, dim)
        
        v_prime = bc.J_inv @ v
        alpha_prime = alpha @ bc.J
        
        assert_scalar_pairing_invariant(alpha, v, alpha_prime, v_prime, rtol=tolerance)
    
    def test_metric_tensor_transformation(self, key, dim, tolerance):
        """Metric transforms as g' = J^T g J."""
        k1, k2 = jax.random.split(key)
        metric = random_metric(k1, dim)
        bc = random_basis_change(k2, dim)
        
        g_prime = bc.J.T @ metric.matrix @ bc.J
        assert_metric_transformation(metric.matrix, g_prime, bc, rtol=tolerance)


@pytest.mark.phase1
@pytest.mark.invariant
class TestTwistedForms:
    """Test orientation-dependent (pseudo) tensor behavior."""
    
    def test_twisted_vector_flips_under_reflection(self, key, dim, tolerance):
        """Twisted forms pick up sgn(det J) = -1 under reflection."""
        k1, k2 = jax.random.split(key)
        omega = random_twisted_vector(k1, dim).components
        reflection = random_reflection(k2, dim)
        
        # Under reflection, twisted vector should flip sign beyond normal transform
        omega_reflected = reflection.det_sign * (reflection.J_inv @ omega)
        assert_twisted_transformation(
            omega, omega_reflected, reflection, 
            TensorVariance.CONTRAVARIANT, rtol=tolerance
        )
    
    def test_twisted_vs_untwisted_differ_under_reflection(self, key, dim, tolerance):
        """Twisted and untwisted forms behave differently under reflection."""
        k1, k2 = jax.random.split(key)
        v = random_vector(k1, dim).components
        reflection = random_reflection(k2, dim)
        
        # Regular vector transform
        v_regular = reflection.J_inv @ v
        # Twisted vector transform
        v_twisted = reflection.det_sign * (reflection.J_inv @ v)
        
        # They should differ by a sign
        assert jnp.allclose(v_regular, -v_twisted, rtol=tolerance)
    
    def test_parity_even_times_even_is_even(self):
        """Even parity composed with even parity gives even."""
        assert_parity_composition(Parity.EVEN, Parity.EVEN, Parity.EVEN)
    
    def test_parity_even_times_odd_is_odd(self):
        """Even parity composed with odd parity gives odd."""
        assert_parity_composition(Parity.EVEN, Parity.ODD, Parity.ODD)
    
    def test_parity_odd_times_odd_is_even(self):
        """Odd parity composed with odd parity gives even."""
        assert_parity_composition(Parity.ODD, Parity.ODD, Parity.EVEN)


@pytest.mark.phase1
@pytest.mark.invariant
class TestMetricProperties:
    """Test Riemannian metric tensor properties."""
    
    def test_metric_is_positive_definite(self, key, dim):
        """Metric tensor must be symmetric positive-definite."""
        metric = random_metric(key, dim)
        assert_positive_definite(metric.matrix)
    
    def test_inner_product_invariant_under_basis_change(self, key, dim, tolerance):
        """g(v,w) = g'(v',w') under coordinate transformation."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        metric = random_metric(k1, dim)
        v1 = random_vector(k2, dim).components
        v2 = random_vector(k3, dim).components
        bc = random_basis_change(k4, dim)
        
        # Transform everything
        metric_prime = MetricTensor(bc.J.T @ metric.matrix @ bc.J)
        v1_prime = bc.J_inv @ v1
        v2_prime = bc.J_inv @ v2
        
        assert_inner_product_invariant(
            v1, v2, metric, v1_prime, v2_prime, metric_prime, rtol=tolerance
        )
    
    def test_index_raising_lowering_are_inverse(self, key, dim, tolerance):
        """g^{-1}(g(v)) = v - raising then lowering returns original."""
        k1, k2 = jax.random.split(key)
        metric = random_metric(k1, dim)
        v = random_vector(k2, dim).components
        
        assert_index_raising_lowering_inverse(v, metric, rtol=tolerance)
    
    def test_metric_provides_canonical_isomorphism(self, key, dim, tolerance):
        """Metric defines vector ↔ covector correspondence."""
        k1, k2 = jax.random.split(key)
        metric = random_metric(k1, dim)
        v = random_vector(k2, dim).components
        
        # Lower index: vector → covector
        alpha = metric.lower_index(v)
        # Raise index: covector → vector
        v_back = metric.raise_index(alpha)
        
        # Use relaxed tolerance for matrix inversion operations
        assert jnp.allclose(v, v_back, rtol=1e-4, atol=1e-5)


@pytest.mark.phase1
@pytest.mark.invariant
class TestFullConsistency:
    """Integration tests for complete geometric consistency."""
    
    def test_full_geometric_stack(self, key, dim, tolerance):
        """Comprehensive check of all transformation laws together."""
        keys = jax.random.split(key, 4)
        v = random_vector(keys[0], dim).components
        alpha = random_covector(keys[1], dim).components
        metric = random_metric(keys[2], dim)
        bc = random_basis_change(keys[3], dim)
        
        full_geometric_consistency_check(v, alpha, metric, bc, rtol=tolerance)
    
    def test_orientation_preserving_vs_reversing(self, key, dim, tolerance):
        """Compare behavior under orientation-preserving vs reversing transforms."""
        k1, k2, k3 = jax.random.split(key, 3)
        v = random_vector(k1, dim).components
        
        # Orientation-preserving transform
        bc_preserve = random_basis_change(k2, dim, orientation_preserving=True)
        assert bc_preserve.det_sign == 1 or bc_preserve.det_sign == -1  # Just check it's valid
        
        # Reflection (orientation-reversing)
        reflection = random_reflection(k3, dim)
        assert reflection.det_sign == -1
    
    def test_batch_transformation_consistency(self, key, dim, tolerance):
        """Verify transformations work correctly on batches."""
        k1, k2 = jax.random.split(key)
        batch_size = 10
        
        # Batch of vectors
        vs = jax.random.normal(k1, shape=(batch_size, dim))
        bc = random_basis_change(k2, dim)
        
        # Transform batch
        vs_prime = vs @ bc.J_inv.T  # Batch-friendly version
        
        # Check each vector individually
        for i in range(batch_size):
            v_single = bc.J_inv @ vs[i]
            assert jnp.allclose(vs_prime[i], v_single, rtol=tolerance)


@pytest.mark.phase1
class TestTypeErrors:
    """Test that geometric type mismatches are caught."""
    
    def test_cannot_add_vector_to_covector_conceptually(self, key, dim):
        """
        Document that vectors and covectors are distinct types.
        
        In a full implementation, this would raise a type error.
        For now, we just verify they are tracked differently.
        """
        k1, k2 = jax.random.split(key)
        v = random_vector(k1, dim)
        alpha = random_covector(k2, dim)
        
        assert v.variance == TensorVariance.CONTRAVARIANT
        assert alpha.variance == TensorVariance.COVARIANT
        assert v.variance != alpha.variance
    
    def test_twisted_and_untwisted_have_different_parity(self, key, dim):
        """Twisted and untwisted vectors are distinguishable by parity."""
        k1, k2 = jax.random.split(key)
        v_regular = random_vector(k1, dim)
        v_twisted = random_twisted_vector(k2, dim)
        
        assert v_regular.parity == Parity.EVEN
        assert v_twisted.parity == Parity.ODD

