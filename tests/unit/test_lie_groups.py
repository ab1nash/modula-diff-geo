"""
Tests for Lie Group Operations (O(1) Rodrigues formula and retractions)

Tests cover:
- SO(3) exp/log inverse property
- Rodrigues formula matches matrix exponential
- Retractions stay on manifold
- Geodesic interpolation
"""
import pytest
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from diffgeo.geometry.lie_groups import (
    skew_symmetric,
    vee,
    so3_exp,
    so3_log,
    so3_geodesic,
    qr_retraction,
    polar_retraction,
    cayley_retraction,
    spd_retraction,
    is_rotation_matrix,
    random_rotation,
    angle_between_rotations,
)


class TestSkewSymmetric:
    """Tests for skew-symmetric utilities."""
    
    def test_skew_symmetric_shape(self):
        """Skew matrix should be 3x3."""
        v = jnp.array([1.0, 2.0, 3.0])
        K = skew_symmetric(v)
        assert K.shape == (3, 3)
    
    def test_skew_symmetric_antisymmetry(self):
        """K^T = -K for skew-symmetric matrix."""
        v = jnp.array([1.0, 2.0, 3.0])
        K = skew_symmetric(v)
        np.testing.assert_allclose(K.T, -K, atol=1e-10)
    
    def test_skew_cross_product(self):
        """[v]× w = v × w."""
        v = jnp.array([1.0, 2.0, 3.0])
        w = jnp.array([4.0, 5.0, 6.0])
        
        K = skew_symmetric(v)
        cross_via_matrix = K @ w
        cross_direct = jnp.cross(v, w)
        
        np.testing.assert_allclose(cross_via_matrix, cross_direct, atol=1e-10)
    
    def test_vee_inverse_of_skew(self):
        """vee([v]×) = v."""
        v = jnp.array([1.0, 2.0, 3.0])
        K = skew_symmetric(v)
        v_recovered = vee(K)
        
        np.testing.assert_allclose(v_recovered, v, atol=1e-10)


class TestSO3ExpLog:
    """Tests for SO(3) exponential and logarithm maps."""
    
    def test_exp_identity_at_zero(self):
        """exp(0) = I."""
        omega = jnp.zeros(3)
        R = so3_exp(omega)
        np.testing.assert_allclose(R, jnp.eye(3), atol=1e-10)
    
    def test_exp_produces_rotation_matrix(self):
        """exp(omega) should be in SO(3)."""
        key = jax.random.PRNGKey(42)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            omega = jax.random.normal(subkey, shape=(3,))
            R = so3_exp(omega)
            
            assert is_rotation_matrix(R), "exp(omega) should be rotation matrix"
    
    def test_log_exp_inverse(self):
        """log(exp(omega)) ≈ omega for small omega."""
        key = jax.random.PRNGKey(42)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            # Use small omega to avoid branch cut issues
            omega = jax.random.normal(subkey, shape=(3,)) * 0.5
            R = so3_exp(omega)
            omega_recovered = so3_log(R)
            
            np.testing.assert_allclose(omega_recovered, omega, atol=1e-5)
    
    def test_exp_log_inverse(self):
        """exp(log(R)) = R for valid rotation matrices."""
        key = jax.random.PRNGKey(123)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            R = random_rotation(subkey)
            omega = so3_log(R)
            R_recovered = so3_exp(omega)
            
            np.testing.assert_allclose(R_recovered, R, atol=1e-5)
    
    def test_rodrigues_matches_matrix_exp(self):
        """Rodrigues formula should match scipy expm for SO(3)."""
        key = jax.random.PRNGKey(456)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            omega = jax.random.normal(subkey, shape=(3,))
            
            # Rodrigues
            R_rodrigues = so3_exp(omega)
            
            # Matrix exponential
            K = skew_symmetric(omega)
            R_expm = jla.expm(K)
            
            np.testing.assert_allclose(R_rodrigues, R_expm, atol=1e-5)
    
    def test_rotation_angle_correct(self):
        """Rotation by omega should have angle ||omega||."""
        omega = jnp.array([0.0, 0.0, jnp.pi / 4])  # 45 degree rotation around z
        R = so3_exp(omega)
        
        # Trace of rotation matrix: 1 + 2*cos(theta)
        trace = jnp.trace(R)
        theta_from_trace = jnp.arccos((trace - 1) / 2)
        
        np.testing.assert_allclose(theta_from_trace, jnp.pi / 4, atol=1e-5)
    
    def test_small_angle_stability(self):
        """Small angle case should be numerically stable."""
        # Very small omega
        omega = jnp.array([1e-8, 1e-9, 1e-10])
        R = so3_exp(omega)
        
        # Should be very close to identity
        np.testing.assert_allclose(R, jnp.eye(3), atol=1e-6)
        assert is_rotation_matrix(R)


class TestSO3Geodesic:
    """Tests for geodesic interpolation on SO(3)."""
    
    def test_geodesic_endpoints(self):
        """gamma(0) = R1, gamma(1) = R2."""
        key = jax.random.PRNGKey(789)
        key1, key2 = jax.random.split(key)
        
        R1 = random_rotation(key1)
        R2 = random_rotation(key2)
        
        np.testing.assert_allclose(so3_geodesic(R1, R2, 0.0), R1, atol=1e-5)
        np.testing.assert_allclose(so3_geodesic(R1, R2, 1.0), R2, atol=1e-5)
    
    def test_geodesic_midpoint_on_manifold(self):
        """Geodesic midpoint should be a rotation matrix."""
        key = jax.random.PRNGKey(101)
        key1, key2 = jax.random.split(key)
        
        R1 = random_rotation(key1)
        R2 = random_rotation(key2)
        
        R_mid = so3_geodesic(R1, R2, 0.5)
        assert is_rotation_matrix(R_mid)
    
    def test_angle_interpolation(self):
        """Geodesic should interpolate angle linearly."""
        R1 = jnp.eye(3)
        omega = jnp.array([0.0, 0.0, 1.0])  # 1 radian around z
        R2 = so3_exp(omega)
        
        for t in [0.25, 0.5, 0.75]:
            R_t = so3_geodesic(R1, R2, t)
            angle = angle_between_rotations(R1, R_t)
            np.testing.assert_allclose(angle, t * 1.0, atol=1e-4)


class TestRetractions:
    """Tests for manifold retractions."""
    
    def test_qr_retraction_orthogonal(self):
        """QR retraction should produce orthogonal matrix."""
        key = jax.random.PRNGKey(111)
        key1, key2 = jax.random.split(key)
        
        base = random_rotation(key1)
        tangent = jax.random.normal(key2, shape=(3, 3)) * 0.1
        
        result = qr_retraction(base, tangent)
        
        # Check orthogonality
        np.testing.assert_allclose(result.T @ result, jnp.eye(3), atol=1e-5)
    
    def test_polar_retraction_orthogonal(self):
        """Polar retraction should produce orthogonal matrix."""
        key = jax.random.PRNGKey(222)
        key1, key2 = jax.random.split(key)
        
        base = random_rotation(key1)
        tangent = jax.random.normal(key2, shape=(3, 3)) * 0.1
        
        result = polar_retraction(base, tangent)
        
        # Check orthogonality
        np.testing.assert_allclose(result.T @ result, jnp.eye(3), atol=1e-5)
    
    def test_cayley_retraction_orthogonal(self):
        """Cayley retraction should produce orthogonal matrix."""
        key = jax.random.PRNGKey(333)
        key1, key2 = jax.random.split(key)
        
        base = random_rotation(key1)
        # Tangent must be in tangent space: skew-symmetric
        omega = jax.random.normal(key2, shape=(3,)) * 0.1
        tangent = skew_symmetric(omega) @ base
        
        result = cayley_retraction(base, tangent)
        
        # Check orthogonality
        np.testing.assert_allclose(result.T @ result, jnp.eye(3), atol=1e-5)
    
    def test_retractions_agree_at_base(self):
        """All retractions should agree when tangent is zero."""
        key = jax.random.PRNGKey(444)
        base = random_rotation(key)
        tangent = jnp.zeros((3, 3))
        
        qr_result = qr_retraction(base, tangent)
        polar_result = polar_retraction(base, tangent)
        
        # Note: Cayley needs compatible tangent, skip for zero
        np.testing.assert_allclose(qr_result, base, atol=1e-5)
        np.testing.assert_allclose(polar_result, base, atol=1e-5)
    
    def test_spd_retraction_positive_definite(self):
        """SPD retraction should produce positive definite matrix."""
        key = jax.random.PRNGKey(555)
        key1, key2 = jax.random.split(key)
        
        # Create SPD base
        A = jax.random.normal(key1, shape=(4, 4))
        base = A @ A.T + 0.1 * jnp.eye(4)
        
        # Random symmetric tangent
        T = jax.random.normal(key2, shape=(4, 4)) * 0.1
        tangent = (T + T.T) / 2
        
        result = spd_retraction(base, tangent)
        
        # Check symmetric
        np.testing.assert_allclose(result, result.T, atol=1e-10)
        
        # Check positive definite (all eigenvalues > 0)
        eigvals = jnp.linalg.eigvalsh(result)
        assert jnp.all(eigvals > 0)


class TestUtilities:
    """Tests for utility functions."""
    
    def test_is_rotation_matrix_true(self):
        """Valid rotation matrix should pass check."""
        R = jnp.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=jnp.float32)
        assert is_rotation_matrix(R)
    
    def test_is_rotation_matrix_false_reflection(self):
        """Reflection (det=-1) should fail check."""
        R = jnp.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=jnp.float32)
        assert not is_rotation_matrix(R)
    
    def test_is_rotation_matrix_false_not_orthogonal(self):
        """Non-orthogonal matrix should fail check."""
        R = jnp.array([
            [1, 0.1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=jnp.float32)
        assert not is_rotation_matrix(R)
    
    def test_random_rotation_valid(self):
        """random_rotation should produce valid rotations."""
        key = jax.random.PRNGKey(666)
        for i in range(10):
            key, subkey = jax.random.split(key)
            R = random_rotation(subkey)
            assert is_rotation_matrix(R)
    
    def test_angle_between_identical_rotations(self):
        """Angle between identical rotations should be 0."""
        key = jax.random.PRNGKey(777)
        R = random_rotation(key)
        angle = angle_between_rotations(R, R)
        np.testing.assert_allclose(angle, 0.0, atol=1e-5)
    
    def test_angle_between_90_degree_rotation(self):
        """90 degree rotation should have angle pi/2."""
        R1 = jnp.eye(3)
        R2 = jnp.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=jnp.float32)
        
        angle = angle_between_rotations(R1, R2)
        np.testing.assert_allclose(angle, jnp.pi / 2, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

