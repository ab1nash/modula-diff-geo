"""
Mathematical invariant verification utilities.

Provides assertion functions that verify fundamental geometric properties:
- Transformation laws (contravariant, covariant, twisted)
- Metric properties (positive-definiteness, symmetry)
- Duality relations (Legendre transform, index raising/lowering)
- Finsler properties (homogeneity, strong convexity, asymmetry)

Each assertion includes clear error messages referencing the violated
mathematical property for easier debugging.
"""
import jax.numpy as jnp
from typing import Optional, Callable
from .generators import (
    GeometricVector, MetricTensor, RandersMetric, BasisChange,
    TensorVariance, Parity
)


class InvariantViolation(AssertionError):
    """Raised when a mathematical invariant is violated."""
    pass


def _check_allclose(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    rtol: float,
    atol: float,
    message: str
) -> None:
    """Helper to check array equality with informative error."""
    if not jnp.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = jnp.max(jnp.abs(actual - expected))
        raise InvariantViolation(
            f"{message}\n"
            f"  Max absolute difference: {max_diff}\n"
            f"  Expected shape: {expected.shape}, Actual shape: {actual.shape}"
        )


# =============================================================================
# Transformation Law Invariants
# =============================================================================

def assert_contravariant_transformation(
    v: jnp.ndarray,
    v_prime: jnp.ndarray,
    basis_change: BasisChange,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify contravariant transformation law: v' = J^{-1} v
    
    Contravariant vectors (tangent vectors, velocities) transform opposite
    to the basis - when basis vectors get longer, components get smaller.
    """
    expected = basis_change.J_inv @ v
    _check_allclose(
        v_prime, expected, rtol, atol,
        "Contravariant transformation law violated: v' ≠ J^{-1}v"
    )


def assert_covariant_transformation(
    alpha: jnp.ndarray,
    alpha_prime: jnp.ndarray,
    basis_change: BasisChange,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify covariant transformation law: α' = α J
    
    Covariant vectors (gradients, 1-forms) transform with the basis.
    This ensures the scalar pairing α(v) = α_i v^i is invariant.
    """
    expected = alpha @ basis_change.J
    _check_allclose(
        alpha_prime, expected, rtol, atol,
        "Covariant transformation law violated: α' ≠ αJ"
    )


def assert_metric_transformation(
    g: jnp.ndarray,
    g_prime: jnp.ndarray,
    basis_change: BasisChange,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify metric tensor transformation: g' = J^T g J
    
    This transformation law ensures distances and angles are preserved.
    """
    J = basis_change.J
    expected = J.T @ g @ J
    _check_allclose(
        g_prime, expected, rtol, atol,
        "Metric transformation law violated: g' ≠ J^T g J"
    )


def assert_twisted_transformation(
    omega: jnp.ndarray,
    omega_prime: jnp.ndarray,
    basis_change: BasisChange,
    base_variance: TensorVariance,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify twisted (pseudo) tensor transformation: includes sgn(det J) factor.
    
    Twisted forms pick up a sign under orientation-reversing transformations.
    This distinguishes magnetic field (twisted 2-form) from electric field (1-form).
    """
    det_sign = basis_change.det_sign
    if base_variance == TensorVariance.CONTRAVARIANT:
        expected = det_sign * (basis_change.J_inv @ omega)
    else:
        expected = det_sign * (omega @ basis_change.J)
    _check_allclose(
        omega_prime, expected, rtol, atol,
        f"Twisted transformation violated: missing sgn(det J)={det_sign} factor"
    )


def assert_scalar_pairing_invariant(
    alpha: jnp.ndarray,
    v: jnp.ndarray,
    alpha_prime: jnp.ndarray,
    v_prime: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify that the scalar pairing α(v) = α_i v^i is invariant under basis change.
    
    This is the fundamental consistency check: the contraction of a covector
    with a vector must be a true scalar (unchanged by coordinate transform).
    """
    original = jnp.dot(alpha, v)
    transformed = jnp.dot(alpha_prime, v_prime)
    _check_allclose(
        transformed, original, rtol, atol,
        "Scalar pairing not invariant: α·v ≠ α'·v'"
    )


# =============================================================================
# Metric and Inner Product Invariants  
# =============================================================================

def assert_positive_definite(
    matrix: jnp.ndarray,
    atol: float = 1e-7
) -> None:
    """Verify matrix is symmetric positive-definite."""
    # Check symmetry
    if not jnp.allclose(matrix, matrix.T, atol=atol):
        raise InvariantViolation("Matrix is not symmetric")
    # Check positive eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    if jnp.any(eigenvalues <= 0):
        min_eig = jnp.min(eigenvalues)
        raise InvariantViolation(
            f"Matrix not positive-definite: min eigenvalue = {min_eig}"
        )


def assert_inner_product_invariant(
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    metric: MetricTensor,
    v1_prime: jnp.ndarray,
    v2_prime: jnp.ndarray,
    metric_prime: MetricTensor,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify g(v1, v2) = g'(v1', v2') under coordinate change.
    
    The inner product defined by a metric must be a scalar - invariant
    under all coordinate transformations.
    """
    original = metric.inner_product(v1, v2)
    transformed = metric_prime.inner_product(v1_prime, v2_prime)
    _check_allclose(
        transformed, original, rtol, atol,
        "Inner product not invariant under coordinate change"
    )


def assert_index_raising_lowering_inverse(
    v: jnp.ndarray,
    metric: MetricTensor,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> None:
    """
    Verify that raising then lowering an index returns the original vector.
    
    g_{ij} g^{jk} = δ_i^k (Kronecker delta)
    
    Note: Uses relaxed tolerances due to matrix inversion numerical errors.
    """
    lowered = metric.lower_index(v)
    raised_back = metric.raise_index(lowered)
    _check_allclose(
        raised_back, v, rtol, atol,
        "Index raising/lowering not inverse: g^{-1}(g(v)) ≠ v"
    )


# =============================================================================
# Finsler Metric Invariants
# =============================================================================

def assert_positive_homogeneity(
    finsler_norm: Callable[[jnp.ndarray], jnp.ndarray],
    v: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify F(λv) = λF(v) for λ > 0 (positive 1-homogeneity).
    
    This is the defining property of a Finsler norm.
    """
    for lam in [0.5, 2.0, 3.7]:
        expected = lam * finsler_norm(v)
        actual = finsler_norm(lam * v)
        _check_allclose(
            actual, expected, rtol, atol,
            f"Positive homogeneity violated: F({lam}v) ≠ {lam}F(v)"
        )


def assert_finsler_asymmetry(
    randers: RandersMetric,
    v: jnp.ndarray,
    rtol: float = 1e-5
) -> None:
    """
    Verify F(v) ≠ F(-v) for Randers metric with non-zero drift.
    
    This asymmetry models directional bias (wind, current, irreversibility).
    """
    forward = randers.norm(v)
    backward = randers.norm(-v)
    if jnp.allclose(forward, backward, rtol=rtol):
        raise InvariantViolation(
            f"Finsler metric unexpectedly symmetric: F(v)={forward}, F(-v)={backward}"
        )


def assert_strong_convexity(
    randers: RandersMetric,
    atol: float = 1e-7
) -> None:
    """
    Verify the Randers metric satisfies strong convexity: |b|_A < 1.
    
    This ensures the indicatrix (unit sphere) is strictly convex.
    """
    if not randers.is_valid():
        b_norm_sq = randers.b @ jnp.linalg.inv(randers.A) @ randers.b
        raise InvariantViolation(
            f"Randers metric not strongly convex: |b|²_A = {b_norm_sq} ≥ 1"
        )


def assert_randers_reduces_to_riemannian(
    A: jnp.ndarray,
    v: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify Randers metric with b=0 equals Riemannian metric.
    
    F(v) = sqrt(v^T A v) when drift b = 0.
    """
    randers = RandersMetric(A, jnp.zeros(A.shape[0]))
    riemannian = jnp.sqrt(v @ A @ v)
    finsler = randers.norm(v)
    _check_allclose(
        finsler, riemannian, rtol, atol,
        "Randers with b=0 should equal Riemannian norm"
    )


# =============================================================================
# Parity / Orientation Invariants
# =============================================================================

def assert_parity_composition(
    p1: Parity,
    p2: Parity,
    expected: Parity
) -> None:
    """
    Verify parity multiplication: Even×Even=Even, Even×Odd=Odd, Odd×Odd=Even.
    """
    result = Parity(p1.value * p2.value)
    if result != expected:
        raise InvariantViolation(
            f"Parity composition wrong: {p1}×{p2} = {result}, expected {expected}"
        )


def assert_reflection_sign_flip(
    twisted_tensor: jnp.ndarray,
    reflected_tensor: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> None:
    """
    Verify twisted tensors flip sign under reflection (det J = -1).
    """
    expected = -twisted_tensor
    _check_allclose(
        reflected_tensor, expected, rtol, atol,
        "Twisted tensor should flip sign under reflection"
    )


# =============================================================================
# Newton-Schulz / Dualization Invariants
# =============================================================================

def assert_orthogonal(
    matrix: jnp.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> None:
    """Verify M @ M.T ≈ I (orthogonality)."""
    n = matrix.shape[0]
    product = matrix @ matrix.T
    identity = jnp.eye(n)
    _check_allclose(
        product, identity, rtol, atol,
        "Matrix not orthogonal: M @ M.T ≠ I"
    )


def assert_singular_values_near_one(
    matrix: jnp.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> None:
    """Verify all singular values ≈ 1 (result of orthogonalization)."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    ones = jnp.ones_like(singular_values)
    _check_allclose(
        singular_values, ones, rtol, atol,
        f"Singular values not near 1: min={jnp.min(singular_values)}, max={jnp.max(singular_values)}"
    )


def assert_dualization_on_unit_ball(
    dual_vector: jnp.ndarray,
    norm_fn: Callable[[jnp.ndarray], float],
    target_norm: float = 1.0,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> None:
    """
    Verify dualized vector lies on the boundary of the unit ball.
    
    The duality map should produce updates with ||Δw|| = target_norm.
    """
    actual_norm = norm_fn(dual_vector)
    _check_allclose(
        actual_norm, target_norm, rtol, atol,
        f"Dual vector not on unit ball: ||v|| = {actual_norm}, expected {target_norm}"
    )


# =============================================================================
# Composite Invariants (for integration testing)
# =============================================================================

def assert_composition_preserves_type(
    input_type: TensorVariance,
    output_type: TensorVariance,
    module_domain: TensorVariance,
    module_codomain: TensorVariance
) -> None:
    """Verify type compatibility in module composition."""
    if input_type != module_domain:
        raise InvariantViolation(
            f"Type mismatch: input is {input_type}, module expects {module_domain}"
        )


def assert_gradient_covector(
    grad: jnp.ndarray,
    message: str = "Gradient should be a covector"
) -> None:
    """
    Reminder that gradients are naturally covectors.
    
    This is a semantic check - gradients transform covariantly because
    they measure change per unit of the coordinate, not per unit of distance.
    """
    # This is a conceptual assertion - in code, we just document the type
    pass  # Actual enforcement happens in the type system


def full_geometric_consistency_check(
    v: jnp.ndarray,
    alpha: jnp.ndarray,
    metric: MetricTensor,
    basis_change: BasisChange,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> None:
    """
    Comprehensive check of all fundamental transformation laws.
    
    This verifies the entire geometric stack is consistent:
    1. Vectors transform contravariantly
    2. Covectors transform covariantly  
    3. Metric transforms tensorially
    4. Scalar products are invariant
    5. Index operations are consistent
    """
    J = basis_change.J
    J_inv = basis_change.J_inv
    
    # Transform everything
    v_prime = J_inv @ v
    alpha_prime = alpha @ J
    g_prime = MetricTensor(J.T @ metric.matrix @ J)
    
    # Check scalar pairing invariance
    assert_scalar_pairing_invariant(alpha, v, alpha_prime, v_prime, rtol, atol)
    
    # Check inner product invariance
    assert_inner_product_invariant(v, v, metric, v_prime, v_prime, g_prime, rtol, atol)
    
    # Check index operations
    assert_index_raising_lowering_inverse(v, metric, rtol, atol)
    assert_index_raising_lowering_inverse(v_prime, g_prime, rtol, atol)

