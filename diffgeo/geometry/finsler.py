"""
Finsler geometry for asymmetric metrics and directed pattern mining.

This module implements Finsler metrics, which generalize Riemannian metrics
to allow asymmetry: the cost to travel from A→B may differ from B→A.

Key classes:
- RandersMetric: F(v) = sqrt(v^T A v) + b^T v (Riemannian + drift)
- FinslerDualizer: Converts Finsler gradients to updates

Applications:
- Directed graph embedding (social networks, citation graphs)
- Causal modeling (time arrow, irreversibility)
- Thermodynamic flows (entropy production)

Reference: Design Document Section 2.2 "Finsler Geometry and Asymmetry"
Reference: Bao-Chern-Shen "An Introduction to Riemann-Finsler Geometry"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional
from functools import partial


class FinslerNorm:
    """
    Abstract base for Finsler norms F(x, v).
    
    A Finsler norm satisfies:
    1. Positive homogeneity: F(λv) = λF(v) for λ > 0
    2. Positive definiteness: F(v) > 0 for v ≠ 0
    3. Strong convexity: The Hessian ∂²(F²)/∂v^i∂v^j is positive-definite
    
    Unlike Riemannian metrics, F(v) ≠ F(-v) in general (asymmetry).
    """
    
    def norm(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute F(v). Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        """Alias for norm()."""
        return self.norm(v)
    
    def is_symmetric(self) -> bool:
        """Check if F(v) = F(-v)."""
        raise NotImplementedError
    
    def dual_norm(self, ell: jnp.ndarray) -> jnp.ndarray:
        """Compute the dual (co-Finsler) norm F*(ℓ) on covectors."""
        raise NotImplementedError


@dataclass
class RandersMetric(FinslerNorm):
    """
    Randers metric: F(v) = sqrt(v^T A v) + b^T v
    
    This is the most tractable Finsler structure. It combines:
    - A: Symmetric positive-definite matrix (Riemannian part)
    - b: Drift vector (asymmetric part, "wind")
    
    Physical interpretation: 
    The Riemannian part measures intrinsic distance. The drift models
    a "current" or "wind" - moving with it is cheaper, against it costly.
    
    Strong convexity condition: |b|_A < 1 where |b|²_A = b^T A^{-1} b.
    
    Attributes:
        A: Symmetric positive-definite (n, n) - Riemannian component
        b: Drift vector (n,) - must satisfy |b|_A < 1
    """
    A: jnp.ndarray  # SPD matrix (n, n)
    b: jnp.ndarray  # Drift vector (n,)
    
    def __post_init__(self):
        """Cache useful quantities."""
        self._A_inv: Optional[jnp.ndarray] = None
        self._b_norm_sq: Optional[float] = None
    
    @property
    def dim(self) -> int:
        return self.A.shape[0]
    
    @property
    def A_inv(self) -> jnp.ndarray:
        """Cached inverse of A."""
        if self._A_inv is None:
            self._A_inv = jnp.linalg.inv(self.A)
        return self._A_inv
    
    @property 
    def b_norm_A_sq(self) -> float:
        """Compute |b|²_A = b^T A^{-1} b."""
        if self._b_norm_sq is None:
            self._b_norm_sq = float(self.b @ self.A_inv @ self.b)
        return self._b_norm_sq
    
    def is_valid(self) -> bool:
        """Check strong convexity: |b|_A < 1."""
        return self.b_norm_A_sq < 1.0
    
    def norm(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute F(v) = sqrt(v^T A v) + b^T v"""
        riemannian_part = jnp.sqrt(v @ self.A @ v)
        drift_part = self.b @ v
        return riemannian_part + drift_part
    
    def is_symmetric(self) -> bool:
        """Randers is symmetric iff b = 0."""
        return jnp.allclose(self.b, 0)
    
    def dual_norm(self, ell: jnp.ndarray) -> jnp.ndarray:
        """Compute dual Randers norm F*(ℓ)."""
        lambda_factor = 1.0 - self.b_norm_A_sq
        ell_shifted = ell
        return jnp.sqrt(ell_shifted @ self.A_inv @ ell_shifted) / jnp.sqrt(lambda_factor)
    
    def gradient_of_norm_sq(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute ∂(F²)/∂v = 2F(v) · ∂F/∂v"""
        norm_A = jnp.sqrt(v @ self.A @ v)
        norm_A_safe = jnp.maximum(norm_A, 1e-8)
        grad_F = (self.A @ v) / norm_A_safe + self.b
        return 2 * self.norm(v) * grad_F
    
    def fundamental_tensor(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute the fundamental tensor g_{ij}(v) = (1/2) ∂²F²/∂v^i∂v^j"""
        norm_A = jnp.sqrt(v @ self.A @ v)
        norm_A_safe = jnp.maximum(norm_A, 1e-8)
        
        Av = self.A @ v
        outer_term = jnp.outer(Av, Av) / (norm_A_safe ** 3)
        g = self.A / norm_A_safe - outer_term
        g = g + self.norm(v) * (self.A / norm_A_safe - outer_term)
        
        return g
    
    @classmethod
    def from_riemannian(cls, A: jnp.ndarray) -> 'RandersMetric':
        """Create Randers metric with zero drift (pure Riemannian)."""
        return cls(A, jnp.zeros(A.shape[0]))
    
    @classmethod
    def with_drift(cls, A: jnp.ndarray, drift_direction: jnp.ndarray, 
                   drift_strength: float = 0.3) -> 'RandersMetric':
        """Create Randers metric with specified drift."""
        assert 0 <= drift_strength < 1, "Drift strength must be in [0, 1)"
        
        A_inv = jnp.linalg.inv(A)
        d_norm = jnp.sqrt(drift_direction @ A_inv @ drift_direction)
        b = drift_direction / d_norm * drift_strength
        
        return cls(A, b)


class FinslerDualizer:
    """
    Converts gradients (covectors) to updates (vectors) for Finsler metrics.
    
    The Finsler duality map J: T*M → TM is defined via:
        J(ℓ) = argmax_{F(v)=1} ℓ(v)
    
    For Randers metrics, this has a tractable form.
    """
    
    def __init__(self, metric: RandersMetric):
        self.metric = metric
    
    def dualize(self, gradient: jnp.ndarray, 
                target_norm: float = 1.0) -> jnp.ndarray:
        """
        Convert gradient (covector) to update vector.
        
        Algorithm for Randers:
        1. Shift gradient by drift component
        2. Apply Riemannian duality (A^{-1})
        3. Rescale to target norm
        """
        shifted_grad = gradient - self.metric.b
        direction = self.metric.A_inv @ shifted_grad
        
        current_norm = self.metric.norm(direction)
        current_norm_safe = jnp.maximum(current_norm, 1e-8)
        
        return direction * (target_norm / current_norm_safe)
    
    def dualize_batch(self, gradients: jnp.ndarray,
                      target_norm: float = 1.0) -> jnp.ndarray:
        """Dualize a batch of gradients."""
        return jax.vmap(partial(self.dualize, target_norm=target_norm))(gradients)


def finsler_orthogonalize(matrix: jnp.ndarray, 
                          drift: jnp.ndarray,
                          n_iters: int = 6) -> jnp.ndarray:
    """
    Orthogonalize matrix accounting for Finsler drift.
    
    Modified Newton-Schulz that biases toward drift-aligned directions.
    """
    ns_coeffs = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    
    transpose = matrix.shape[1] > matrix.shape[0]
    if transpose:
        matrix = matrix.T
    
    # Pre-condition with drift
    drift_magnitude = jnp.linalg.norm(drift)
    if drift_magnitude > 1e-8:
        drift_unit = drift / drift_magnitude
        drift_bias = 0.1 * jnp.outer(
            drift_unit[:matrix.shape[0]], 
            drift_unit[:matrix.shape[1]] if matrix.shape[1] <= len(drift_unit) 
            else jnp.zeros(matrix.shape[1])
        )
        matrix = matrix + drift_bias[:matrix.shape[0], :matrix.shape[1]]
    
    # Standard Newton-Schulz
    matrix = matrix / jnp.linalg.norm(matrix)
    for a, b, c in ns_coeffs:
        gram = matrix.T @ matrix
        identity = jnp.eye(gram.shape[0])
        matrix = matrix @ (a * identity + b * gram + c * gram @ gram)
    
    if transpose:
        matrix = matrix.T
    
    return matrix


def randers_geodesic_distance(
    p1: jnp.ndarray, 
    p2: jnp.ndarray,
    randers: RandersMetric
) -> jnp.ndarray:
    """Approximate geodesic distance in Randers metric."""
    displacement = p2 - p1
    return randers.norm(displacement)


def make_randers_spd(dim: int, key: jax.Array, 
                     drift_strength: float = 0.3) -> RandersMetric:
    """Generate a random valid Randers metric."""
    k1, k2 = jax.random.split(key)
    
    L = jax.random.normal(k1, shape=(dim, dim))
    A = L @ L.T + 0.1 * jnp.eye(dim)
    
    b_raw = jax.random.normal(k2, shape=(dim,))
    A_inv = jnp.linalg.inv(A)
    b_norm = jnp.sqrt(b_raw @ A_inv @ b_raw)
    b = b_raw * (drift_strength / (b_norm + 1e-8))
    
    return RandersMetric(A, b)


__all__ = [
    'FinslerNorm',
    'RandersMetric', 
    'FinslerDualizer',
    'finsler_orthogonalize',
    'randers_geodesic_distance',
    'make_randers_spd',
]

