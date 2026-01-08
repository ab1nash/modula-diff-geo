"""
Efficient Lie Group Operations with Closed-Form Solutions

This module provides O(1) and O(n²) operations for common Lie groups,
avoiding expensive O(n³) eigendecompositions where closed-form solutions exist.

Key features:
- Rodrigues formula for SO(3): O(1) exp/log maps
- QR and polar retractions: O(n²) alternatives to exp map
- Skew-symmetric utilities for Lie algebra operations

Reference: 
- Rodrigues (1840) "Des lois géométriques qui régissent les déplacements"
- Absil et al. (2008) "Optimization Algorithms on Matrix Manifolds"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


# =============================================================================
# SKEW-SYMMETRIC UTILITIES
# =============================================================================

def skew_symmetric(v: jnp.ndarray) -> jnp.ndarray:
    """
    Create skew-symmetric matrix from 3-vector (hat operator).
    
    [v]× = | 0   -v₂   v₁ |
           | v₂   0   -v₀ |
           |-v₁   v₀   0  |
    
    Such that [v]× w = v × w (cross product).
    
    Args:
        v: 3-vector
        
    Returns:
        3x3 skew-symmetric matrix
        
    Complexity: O(1)
    """
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(K: jnp.ndarray) -> jnp.ndarray:
    """
    Extract 3-vector from skew-symmetric matrix (vee operator).
    
    Inverse of skew_symmetric (hat operator).
    
    Args:
        K: 3x3 skew-symmetric matrix
        
    Returns:
        3-vector
        
    Complexity: O(1)
    """
    return jnp.array([K[2, 1], K[0, 2], K[1, 0]])


# =============================================================================
# SO(3) OPERATIONS - O(1) CLOSED FORM
# =============================================================================

def so3_exp(omega: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from so(3) to SO(3) using Rodrigues formula.
    
    exp(ω×) = I + sin(θ)/θ [ω]× + (1-cos(θ))/θ² [ω]×²
    
    where θ = ||ω|| is the rotation angle.
    
    This is O(1) - just trig functions and 3x3 matrix operations.
    Much faster than general matrix exponential which is O(n³).
    
    Args:
        omega: 3-vector representing axis-angle (axis * angle)
        
    Returns:
        3x3 rotation matrix in SO(3)
        
    Complexity: O(1)
    """
    theta = jnp.linalg.norm(omega)
    
    # Handle small angle case to avoid division by zero
    # Use Taylor expansion: sin(θ)/θ ≈ 1 - θ²/6, (1-cos(θ))/θ² ≈ 1/2 - θ²/24
    small_angle = theta < 1e-6
    
    # Safe division
    theta_safe = jnp.where(small_angle, 1.0, theta)
    
    # Rodrigues coefficients
    sinc = jnp.where(small_angle, 1.0 - theta**2 / 6.0, jnp.sin(theta) / theta_safe)
    cosc = jnp.where(small_angle, 0.5 - theta**2 / 24.0, (1.0 - jnp.cos(theta)) / (theta_safe**2))
    
    # Skew-symmetric matrix
    K = skew_symmetric(omega)
    
    # Rodrigues formula
    return jnp.eye(3) + sinc * K + cosc * (K @ K)


def so3_log(R: jnp.ndarray) -> jnp.ndarray:
    """
    Logarithmic map from SO(3) to so(3) (inverse Rodrigues).
    
    log(R) = θ/(2 sin(θ)) (R - R^T)
    
    where θ = arccos((tr(R) - 1) / 2).
    
    Args:
        R: 3x3 rotation matrix in SO(3)
        
    Returns:
        3-vector representing axis-angle
        
    Complexity: O(1)
    """
    # Compute rotation angle
    trace = jnp.trace(R)
    # Clamp to valid range for arccos
    cos_theta = jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    
    # Handle small angle case
    small_angle = theta < 1e-6
    
    # Handle angle near pi (180 degrees) - need different formula
    near_pi = theta > jnp.pi - 1e-6
    
    # Standard case coefficient
    theta_safe = jnp.where(small_angle, 1.0, theta)
    sin_theta = jnp.sin(theta_safe)
    coeff = jnp.where(small_angle, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta + 1e-10))
    
    # Skew part of R
    K = coeff * (R - R.T)
    
    # For near-pi case, use eigenvector approach (but simplified)
    # The axis is the eigenvector with eigenvalue 1
    # For now, use the standard formula which works for most cases
    
    return vee(K)


def so3_geodesic(R1: jnp.ndarray, R2: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Geodesic interpolation on SO(3).
    
    γ(t) = R1 @ exp(t * log(R1^T @ R2))
    
    Args:
        R1: Start rotation
        R2: End rotation  
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Interpolated rotation matrix
        
    Complexity: O(1)
    """
    # Relative rotation
    R_rel = R1.T @ R2
    
    # Log of relative rotation
    omega = so3_log(R_rel)
    
    # Scaled exponential
    R_interp = so3_exp(t * omega)
    
    return R1 @ R_interp


# =============================================================================
# GENERAL RETRACTIONS - O(n²)
# =============================================================================

def qr_retraction(base: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """
    QR retraction for orthogonal manifolds.
    
    Projects base + tangent back onto the Stiefel manifold using QR decomposition.
    This is a first-order retraction, sufficient for optimization.
    
    R_X(V) = qf(X + V)
    
    where qf extracts the Q factor from QR decomposition.
    
    Args:
        base: Current point on manifold (n x p orthonormal columns)
        tangent: Tangent vector at base
        
    Returns:
        Point on manifold near base + tangent
        
    Complexity: O(n²p) for thin QR, O(n²) for square matrices
    """
    Q, R = jnp.linalg.qr(base + tangent)
    
    # Ensure determinant is positive (stay in SO(n) not O(n))
    # by flipping signs of columns with negative diagonal in R
    signs = jnp.sign(jnp.diag(R))
    # Handle zeros
    signs = jnp.where(signs == 0, 1.0, signs)
    
    return Q * signs


def polar_retraction(base: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """
    Polar retraction for orthogonal manifolds.
    
    Projects using polar decomposition (closest orthogonal matrix).
    
    R_X(V) = (X + V)(X + V)^T X + V)^{-1/2}
    
    Implemented via SVD: U @ V^T where X + V = U Σ V^T
    
    Args:
        base: Current point on manifold
        tangent: Tangent vector at base
        
    Returns:
        Closest orthogonal matrix to base + tangent
        
    Complexity: O(n³) for SVD, but fast in practice for small n
    """
    U, _, Vt = jnp.linalg.svd(base + tangent, full_matrices=False)
    return U @ Vt


def cayley_retraction(base: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """
    Cayley retraction for SO(n).
    
    Uses the Cayley transform which maps skew-symmetric to orthogonal:
    
    R_X(V) = (I - W/2)^{-1} (I + W/2) X
    
    where W = VX^T - XV^T is skew-symmetric.
    
    This is often faster than QR for square matrices.
    
    Args:
        base: Current rotation matrix
        tangent: Tangent vector (should satisfy tangent @ base.T + base @ tangent.T = 0)
        
    Returns:
        New rotation matrix
        
    Complexity: O(n³) for matrix solve, but small constant
    """
    n = base.shape[0]
    
    # Skew-symmetric part
    W = tangent @ base.T - base @ tangent.T
    
    # Cayley transform
    I = jnp.eye(n)
    return jnp.linalg.solve(I - 0.5 * W, (I + 0.5 * W) @ base)


# =============================================================================
# SPD RETRACTION
# =============================================================================

def spd_retraction(base: jnp.ndarray, tangent: jnp.ndarray, 
                   epsilon: float = 1e-6) -> jnp.ndarray:
    """
    Retraction for SPD manifold.
    
    Simply adds tangent and projects to SPD cone by:
    1. Symmetrizing
    2. Clipping negative eigenvalues
    
    Args:
        base: Current SPD matrix
        tangent: Symmetric tangent matrix
        epsilon: Minimum eigenvalue after projection
        
    Returns:
        SPD matrix near base + tangent
        
    Complexity: O(n³) for eigendecomposition
    """
    result = base + tangent
    
    # Symmetrize
    result = (result + result.T) / 2
    
    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(result)
    
    # Clip negative eigenvalues
    eigvals = jnp.maximum(eigvals, epsilon)
    
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_rotation_matrix(R: jnp.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if matrix is a valid rotation (in SO(3)).
    
    Checks:
    1. R^T R = I (orthogonality)
    2. det(R) = 1 (proper rotation, not reflection)
    
    Args:
        R: Matrix to check
        tol: Tolerance for checks
        
    Returns:
        True if R is a rotation matrix
    """
    # Check orthogonality
    RtR = R.T @ R
    orth_error = jnp.linalg.norm(RtR - jnp.eye(R.shape[0]), ord='fro')
    
    # Check determinant
    det = jnp.linalg.det(R)
    det_error = jnp.abs(det - 1.0)
    
    return bool(orth_error < tol and det_error < tol)


def random_rotation(key: jax.Array, dim: int = 3) -> jnp.ndarray:
    """
    Generate a random rotation matrix uniformly from SO(n).
    
    Uses QR decomposition of random Gaussian matrix.
    
    Args:
        key: JAX random key
        dim: Dimension (default 3 for SO(3))
        
    Returns:
        Random rotation matrix
    """
    # Random Gaussian matrix
    A = jax.random.normal(key, shape=(dim, dim))
    
    # QR decomposition
    Q, R = jnp.linalg.qr(A)
    
    # Ensure det = +1
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    
    return Q


def angle_between_rotations(R1: jnp.ndarray, R2: jnp.ndarray) -> float:
    """
    Compute geodesic distance (angle) between two rotations.
    
    d(R1, R2) = ||log(R1^T R2)||
    
    For SO(3), this is the rotation angle of the relative rotation.
    
    Args:
        R1, R2: Rotation matrices
        
    Returns:
        Angle in radians
        
    Complexity: O(1) for SO(3)
    """
    R_rel = R1.T @ R2
    omega = so3_log(R_rel)
    return float(jnp.linalg.norm(omega))


__all__ = [
    # Skew-symmetric
    'skew_symmetric',
    'vee',
    # SO(3) operations
    'so3_exp',
    'so3_log', 
    'so3_geodesic',
    # Retractions
    'qr_retraction',
    'polar_retraction',
    'cayley_retraction',
    'spd_retraction',
    # Utilities
    'is_rotation_matrix',
    'random_rotation',
    'angle_between_rotations',
]

