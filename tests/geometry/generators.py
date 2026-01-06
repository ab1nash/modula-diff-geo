"""
Random generators for geometric objects.

Provides factories for creating test instances of vectors, covectors, metrics,
and transformations with known mathematical properties. These generators form
the basis for property-based testing of geometric invariants.

Mathematical Context:
- Vectors (contravariant): transform as v' = J^{-1}v under basis change J
- Covectors (covariant): transform as α' = αJ under basis change J  
- Metric tensors: g' = J^T g J preserves inner products
- Twisted forms: pick up sgn(det J) under orientation-reversing transforms
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum


class TensorVariance(Enum):
    """Classification of tensor transformation behavior."""
    CONTRAVARIANT = "contra"  # Transforms with J^{-1} (vectors, velocities)
    COVARIANT = "co"          # Transforms with J (gradients, 1-forms)
    MIXED = "mixed"           # Has both upper and lower indices


class Parity(Enum):
    """Orientation behavior under reflection."""
    EVEN = 1   # Invariant under reflection (scalars, true tensors)
    ODD = -1   # Sign flip under reflection (pseudotensors, twisted forms)


@dataclass
class GeometricVector:
    """A vector with explicit variance and parity metadata."""
    components: jnp.ndarray
    variance: TensorVariance
    parity: Parity = Parity.EVEN
    
    @property
    def dim(self) -> int:
        return self.components.shape[-1]


@dataclass  
class MetricTensor:
    """Symmetric positive-definite metric tensor."""
    matrix: jnp.ndarray  # Shape (n, n), symmetric positive-definite
    
    @property
    def inverse(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.matrix)
    
    def raise_index(self, covector: jnp.ndarray) -> jnp.ndarray:
        """Convert covector to vector: v^i = g^{ij} α_j"""
        return self.inverse @ covector
    
    def lower_index(self, vector: jnp.ndarray) -> jnp.ndarray:
        """Convert vector to covector: α_i = g_{ij} v^j"""
        return self.matrix @ vector
    
    def inner_product(self, v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
        """Compute g(v1, v2) = v1^i g_{ij} v2^j"""
        return v1 @ self.matrix @ v2


@dataclass
class RandersMetric:
    """
    Finsler metric of Randers type: F(v) = sqrt(v^T A v) + b^T v
    
    Encodes asymmetric "cost" - the drift vector b creates directional bias,
    like wind affecting travel time. Key for modeling irreversible processes.
    """
    A: jnp.ndarray  # Symmetric positive-definite (n, n)
    b: jnp.ndarray  # Drift vector (n,), must satisfy |b|_A < 1
    
    def norm(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute F(v) = sqrt(v^T A v) + b^T v"""
        riemannian_part = jnp.sqrt(v @ self.A @ v)
        drift_part = self.b @ v
        return riemannian_part + drift_part
    
    def is_valid(self) -> bool:
        """Check strong convexity condition: |b|_A < 1"""
        b_norm_sq = self.b @ jnp.linalg.inv(self.A) @ self.b
        return b_norm_sq < 1.0


class BasisChange(NamedTuple):
    """An invertible linear transformation representing a change of basis."""
    J: jnp.ndarray        # Jacobian matrix (n, n)
    J_inv: jnp.ndarray    # Inverse Jacobian
    det_sign: int         # Sign of determinant (+1 or -1)


# =============================================================================
# Generator Functions
# =============================================================================

def random_vector(key: jax.Array, dim: int, dtype=jnp.float32) -> GeometricVector:
    """Generate a random contravariant vector."""
    components = jax.random.normal(key, shape=(dim,), dtype=dtype)
    return GeometricVector(components, TensorVariance.CONTRAVARIANT)


def random_covector(key: jax.Array, dim: int, dtype=jnp.float32) -> GeometricVector:
    """Generate a random covariant vector (1-form)."""
    components = jax.random.normal(key, shape=(dim,), dtype=dtype)
    return GeometricVector(components, TensorVariance.COVARIANT)


def random_twisted_vector(key: jax.Array, dim: int, dtype=jnp.float32) -> GeometricVector:
    """Generate a random twisted (pseudo) vector."""
    components = jax.random.normal(key, shape=(dim,), dtype=dtype)
    return GeometricVector(components, TensorVariance.CONTRAVARIANT, Parity.ODD)


def random_spd_matrix(key: jax.Array, dim: int, dtype=jnp.float32) -> jnp.ndarray:
    """
    Generate random symmetric positive-definite matrix.
    
    Uses A = L @ L.T + εI construction for guaranteed positive-definiteness.
    """
    L = jax.random.normal(key, shape=(dim, dim), dtype=dtype)
    return L @ L.T + 0.1 * jnp.eye(dim, dtype=dtype)


def random_metric(key: jax.Array, dim: int, dtype=jnp.float32) -> MetricTensor:
    """Generate a random Riemannian metric tensor."""
    matrix = random_spd_matrix(key, dim, dtype)
    return MetricTensor(matrix)


def random_randers_metric(
    key: jax.Array, 
    dim: int, 
    drift_strength: float = 0.3,
    dtype=jnp.float32
) -> RandersMetric:
    """
    Generate a random Randers (Finsler) metric.
    
    Args:
        drift_strength: Controls asymmetry, must be < 1 for strong convexity.
    """
    k1, k2 = jax.random.split(key)
    A = random_spd_matrix(k1, dim, dtype)
    # Generate b with controlled norm to ensure |b|_A < 1
    b_raw = jax.random.normal(k2, shape=(dim,), dtype=dtype)
    b_raw = b_raw / jnp.linalg.norm(b_raw)
    # Scale so that b^T A^{-1} b < 1
    A_inv = jnp.linalg.inv(A)
    scale = drift_strength / jnp.sqrt(b_raw @ A_inv @ b_raw + 1e-8)
    b = b_raw * scale
    return RandersMetric(A, b)


def random_orthogonal_matrix(key: jax.Array, dim: int, dtype=jnp.float32) -> jnp.ndarray:
    """Generate random orthogonal matrix via QR decomposition."""
    A = jax.random.normal(key, shape=(dim, dim), dtype=dtype)
    Q, _ = jnp.linalg.qr(A)
    return Q


def random_basis_change(
    key: jax.Array, 
    dim: int,
    orientation_preserving: bool = True,
    dtype=jnp.float32
) -> BasisChange:
    """
    Generate a random invertible basis change transformation.
    
    Uses well-conditioned matrices for numerical stability.
    
    Args:
        orientation_preserving: If False, may flip orientation (det < 0).
    """
    k1, k2 = jax.random.split(key)
    # Start with orthogonal for numerical stability
    J = random_orthogonal_matrix(k1, dim, dtype)
    # Add moderate stretch (avoid extreme condition numbers)
    scales = jax.random.uniform(k2, shape=(dim,), minval=0.7, maxval=1.4, dtype=dtype)
    J = J * scales
    
    if not orientation_preserving:
        # Possibly flip orientation
        flip = jax.random.bernoulli(k2)
        J = J.at[:, 0].set(jnp.where(flip, -J[:, 0], J[:, 0]))
    
    det_sign = jnp.sign(jnp.linalg.det(J)).astype(int)
    J_inv = jnp.linalg.inv(J)
    return BasisChange(J, J_inv, int(det_sign))


def random_reflection(key: jax.Array, dim: int, dtype=jnp.float32) -> BasisChange:
    """Generate a random reflection (orientation-reversing orthogonal transform)."""
    Q = random_orthogonal_matrix(key, dim, dtype)
    # Flip first column to make det = -1
    R = Q.at[:, 0].set(-Q[:, 0])
    return BasisChange(R, R.T, -1)


def random_contact_form(key: jax.Array, dim: int, dtype=jnp.float32) -> jnp.ndarray:
    """
    Generate coefficients of a contact 1-form on R^{2n+1}.
    
    A contact form α satisfies α ∧ (dα)^n ≠ 0 (maximally non-integrable).
    Standard example: α = dz - Σ y_i dx_i on R^{2n+1}.
    """
    assert dim % 2 == 1, "Contact manifolds are odd-dimensional"
    # Return coefficients in standard contact coordinates
    n = (dim - 1) // 2
    # α = dz - y_1 dx_1 - ... - y_n dx_n
    # Perturb slightly for generality
    alpha = jax.random.normal(key, shape=(dim,), dtype=dtype)
    alpha = alpha / jnp.linalg.norm(alpha)
    return alpha


# =============================================================================
# Batch Generators (for property-based testing)
# =============================================================================

def generate_vector_batch(
    key: jax.Array, 
    batch_size: int, 
    dim: int,
    dtype=jnp.float32
) -> jnp.ndarray:
    """Generate batch of random vectors, shape (batch_size, dim)."""
    return jax.random.normal(key, shape=(batch_size, dim), dtype=dtype)


def generate_transform_batch(
    key: jax.Array,
    batch_size: int,
    dim: int,
    dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate batch of random basis changes and their inverses."""
    keys = jax.random.split(key, batch_size)
    
    def make_transform(k):
        bc = random_basis_change(k, dim, dtype=dtype)
        return bc.J, bc.J_inv
    
    Js, J_invs = jax.vmap(make_transform)(keys)
    return Js, J_invs

