"""
Symmetric Positive Definite (SPD) Manifold Operations

Implements Riemannian geometry on the manifold of SPD matrices, with
applications to covariance matrix classification, DTI, and BCI.

From the research document Section 4.2:
"The natural distance between two covariance matrices A and B is given by
the Riemannian metric: d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F"

Key properties:
- Affine invariance: d(WAW^T, WBW^T) = d(A, B)
- No swelling effect (unlike Euclidean mean)
- Geodesics stay in SPD cone

Reference: Pennec, X. et al. (2006). "A Riemannian Framework for Tensor Computing"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


def _matrix_sqrt(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)  # Ensure positivity
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt_inv(A: jnp.ndarray) -> jnp.ndarray:
    """Compute inverse matrix square root."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T


def _matrix_log(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix logarithm for SPD matrix."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T


def _matrix_exp(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix exponential for symmetric matrix."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    return eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T


def _matrix_power(A: jnp.ndarray, t: float) -> jnp.ndarray:
    """Compute A^t for SPD matrix A."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(jnp.power(eigvals, t)) @ eigvecs.T


class SPDManifold:
    """
    Manifold of Symmetric Positive Definite matrices with affine-invariant metric.
    
    The space P_n of n×n SPD matrices forms a Riemannian manifold with
    negative curvature. The affine-invariant metric is:
        <V, W>_P = tr(P^{-1} V P^{-1} W)
    
    This metric is invariant under congruence: d(APA^T, AQA^T) = d(P, Q)
    for any invertible A.
    
    Applications:
    - EEG/BCI: Classify brain states via covariance matrices
    - DTI: Analyze diffusion tensors in brain imaging
    - Radar: Space-time adaptive processing
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Matrix dimension (n×n SPD matrices)
        """
        self.dim = dim
        self.matrix_dim = (dim, dim)
    
    def is_spd(self, A: jnp.ndarray, tol: float = 1e-8) -> bool:
        """
        Check if matrix is SPD (Symmetric Positive Definite).
        
        A matrix is SPD if:
        1. It's symmetric: A = A^T
        2. All eigenvalues are positive
        """
        # Check symmetry
        is_symmetric = jnp.allclose(A, A.T, atol=tol)
        
        # Check positive eigenvalues
        eigvals = jnp.linalg.eigvalsh(A)
        is_positive = jnp.all(eigvals > -tol)
        
        return bool(is_symmetric and is_positive)
    
    def project_to_spd(self, A: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
        """
        Project a matrix to SPD cone.
        
        Makes A symmetric and clips negative eigenvalues.
        """
        # Symmetrize
        A_sym = (A + A.T) / 2
        
        # Clip eigenvalues
        eigvals, eigvecs = jnp.linalg.eigh(A_sym)
        eigvals = jnp.maximum(eigvals, epsilon)
        
        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    
    def distance(self, A: jnp.ndarray, B: jnp.ndarray) -> float:
        """
        Compute affine-invariant Riemannian distance.
        
        d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
                  = sqrt(Σ log²(λ_i))
        
        where λ_i are generalized eigenvalues of (B, A).
        """
        A_inv_sqrt = _matrix_sqrt_inv(A)
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        log_inner = _matrix_log(inner)
        return float(jnp.linalg.norm(log_inner, ord='fro'))
    
    def distance_squared(self, A: jnp.ndarray, B: jnp.ndarray) -> float:
        """Squared Riemannian distance (avoids sqrt)."""
        A_inv_sqrt = _matrix_sqrt_inv(A)
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        log_inner = _matrix_log(inner)
        return float(jnp.sum(log_inner ** 2))
    
    def geodesic(self, A: jnp.ndarray, B: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Compute point along geodesic from A to B at time t ∈ [0,1].
        
        γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}
        
        Properties:
        - γ(0) = A
        - γ(1) = B
        - γ(t) ∈ SPD for all t
        """
        A_sqrt = _matrix_sqrt(A)
        A_inv_sqrt = _matrix_sqrt_inv(A)
        
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        inner_power_t = _matrix_power(inner, t)
        
        return A_sqrt @ inner_power_t @ A_sqrt
    
    def log_map(self, P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
        """
        Logarithmic map: SPD → Tangent space at P.
        
        Log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
        
        Maps Q to a tangent vector at P pointing toward Q.
        The result is a symmetric matrix in the tangent space.
        """
        P_sqrt = _matrix_sqrt(P)
        P_inv_sqrt = _matrix_sqrt_inv(P)
        
        inner = P_inv_sqrt @ Q @ P_inv_sqrt
        log_inner = _matrix_log(inner)
        
        return P_sqrt @ log_inner @ P_sqrt
    
    def exp_map(self, P: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        """
        Exponential map: Tangent space at P → SPD.
        
        Exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
        
        Maps a tangent vector V at P to a point on the manifold.
        """
        P_sqrt = _matrix_sqrt(P)
        P_inv_sqrt = _matrix_sqrt_inv(P)
        
        inner = P_inv_sqrt @ V @ P_inv_sqrt
        exp_inner = _matrix_exp(inner)
        
        return P_sqrt @ exp_inner @ P_sqrt
    
    def parallel_transport(self, 
                           V: jnp.ndarray, 
                           A: jnp.ndarray, 
                           B: jnp.ndarray) -> jnp.ndarray:
        """
        Parallel transport tangent vector V from T_A to T_B along geodesic.
        
        Γ_{A→B}(V) = E V E^T
        
        where E = (BA^{-1})^{1/2}
        
        This preserves the Riemannian norm of V.
        """
        A_inv = jnp.linalg.inv(A)
        E = _matrix_sqrt(B @ A_inv)
        return E @ V @ E.T
    
    def frechet_mean(self, 
                     matrices: List[jnp.ndarray],
                     weights: Optional[jnp.ndarray] = None,
                     max_iter: int = 100,
                     tol: float = 1e-6) -> jnp.ndarray:
        """
        Compute weighted Fréchet mean (Riemannian center of mass).
        
        M* = argmin_M Σ w_i d²(M, P_i)
        
        Uses iterative algorithm:
        1. Compute tangent vectors at current estimate
        2. Average in tangent space
        3. Map back to manifold
        
        Unlike arithmetic mean, preserves determinant and anisotropy.
        """
        n = len(matrices)
        if weights is None:
            weights = jnp.ones(n) / n
        else:
            weights = weights / jnp.sum(weights)
        
        # Initialize with weighted Euclidean mean (projected to SPD)
        M = sum(w * P for w, P in zip(weights, matrices))
        M = self.project_to_spd(M)
        
        for _ in range(max_iter):
            # Compute weighted average in tangent space
            tangent_sum = jnp.zeros_like(M)
            for w, P in zip(weights, matrices):
                tangent_sum += w * self.log_map(M, P)
            
            # Check convergence
            tangent_norm = jnp.linalg.norm(tangent_sum, ord='fro')
            if tangent_norm < tol:
                break
            
            # Update via exponential map
            M = self.exp_map(M, tangent_sum)
        
        return M
    
    def log_euclidean_mean(self, matrices: List[jnp.ndarray],
                           weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute Log-Euclidean mean (faster approximation).
        
        M_LE = exp(Σ w_i log(P_i))
        
        This is not the true Fréchet mean but is much faster to compute
        and often gives similar results.
        """
        n = len(matrices)
        if weights is None:
            weights = jnp.ones(n) / n
        
        log_mean = sum(w * _matrix_log(P) for w, P in zip(weights, matrices))
        return _matrix_exp(log_mean)
    
    def tangent_space_projection(self,
                                 matrices: List[jnp.ndarray],
                                 reference: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Project matrices to tangent space at reference point.
        
        This is the key operation for Riemannian classification:
        1. Compute Fréchet mean of training data
        2. Project all matrices to tangent space at mean
        3. Apply standard Euclidean classifier in tangent space
        
        Returns:
            Flattened tangent vectors suitable for classification
        """
        if reference is None:
            reference = self.frechet_mean(matrices)
        
        tangent_vectors = []
        for P in matrices:
            V = self.log_map(reference, P)
            # Take upper triangle (symmetric matrix is redundant)
            upper_idx = jnp.triu_indices(self.dim)
            v_flat = V[upper_idx]
            tangent_vectors.append(v_flat)
        
        return jnp.stack(tangent_vectors)


@dataclass
class SPDMetricTensor:
    """
    Riemannian metric tensor on SPD manifold at a specific point.
    
    The metric at P ∈ SPD is:
        <V, W>_P = tr(P^{-1} V P^{-1} W)
    """
    base_point: jnp.ndarray  # P ∈ SPD
    
    def __post_init__(self):
        self._P_inv = jnp.linalg.inv(self.base_point)
    
    def inner_product(self, V: jnp.ndarray, W: jnp.ndarray) -> float:
        """Compute <V, W>_P = tr(P^{-1} V P^{-1} W)."""
        return float(jnp.trace(self._P_inv @ V @ self._P_inv @ W))
    
    def norm(self, V: jnp.ndarray) -> float:
        """Compute ||V||_P = sqrt(<V,V>_P)."""
        return float(jnp.sqrt(self.inner_product(V, V)))


class SPDClassifier:
    """
    Riemannian classifier for SPD matrices.
    
    Implements Minimum Distance to Riemannian Mean (MDRM) classifier,
    commonly used in BCI applications.
    """
    
    def __init__(self, manifold: SPDManifold):
        self.manifold = manifold
        self.class_means: dict = {}
    
    def fit(self, matrices: List[jnp.ndarray], labels: jnp.ndarray) -> 'SPDClassifier':
        """
        Fit classifier by computing class-wise Fréchet means.
        """
        unique_labels = jnp.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            class_matrices = [m for m, is_class in zip(matrices, mask) if is_class]
            self.class_means[int(label)] = self.manifold.frechet_mean(class_matrices)
        
        return self
    
    def predict(self, matrix: jnp.ndarray) -> int:
        """
        Predict class by minimum Riemannian distance to class means.
        """
        min_dist = float('inf')
        best_label = -1
        
        for label, mean in self.class_means.items():
            dist = self.manifold.distance(matrix, mean)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        return best_label
    
    def predict_batch(self, matrices: List[jnp.ndarray]) -> jnp.ndarray:
        """Predict labels for a batch of matrices."""
        return jnp.array([self.predict(m) for m in matrices])


__all__ = [
    'SPDManifold',
    'SPDMetricTensor',
    'SPDClassifier',
    '_matrix_sqrt',
    '_matrix_sqrt_inv',
    '_matrix_log',
    '_matrix_exp',
    '_matrix_power',
]

