"""
Metric tensor and geometric vector abstractions.

This module provides:
- MetricTensor: Riemannian metric for index raising/lowering
- GeometricVector: Vector with explicit variance and parity metadata

The metric tensor is the key to converting between vectors and covectors.
In optimization, gradients (covectors) must be converted to updates (vectors)
via the metric: Δw = g^{-1}(∇L).

Reference: Burke "Applied Differential Geometry"
"""
from __future__ import annotations

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

from .core import TensorVariance, Parity


@dataclass
class MetricTensor:
    """
    Riemannian metric tensor g_{ij} for index operations.
    
    The metric provides the canonical isomorphism between vectors and covectors:
    - lower_index: v^i → α_i = g_{ij} v^j (vector to covector)
    - raise_index: α_i → v^i = g^{ij} α_j (covector to vector)
    
    This is precisely what's needed for gradient descent: the gradient ∇L
    lives in the cotangent space (covectors), but weight updates must be
    vectors. The metric converts: Δw = g^{-1}(∇L).
    
    For Euclidean space, g = I, and the conversion is trivial (metric blindness).
    For curved/anisotropic spaces, the metric is essential.
    """
    matrix: jnp.ndarray  # Shape (n, n), symmetric positive-definite
    
    def __post_init__(self):
        """Cache the inverse for efficiency."""
        self._inverse: Optional[jnp.ndarray] = None
    
    @property
    def dim(self) -> int:
        return self.matrix.shape[0]
    
    @property
    def inverse(self) -> jnp.ndarray:
        """Compute and cache the inverse metric g^{ij}."""
        if self._inverse is None:
            self._inverse = jnp.linalg.inv(self.matrix)
        return self._inverse
    
    def lower_index(self, vector: jnp.ndarray) -> jnp.ndarray:
        """
        Convert vector to covector: α_i = g_{ij} v^j
        
        This is "index lowering" - the musical isomorphism ♭ (flat).
        """
        return self.matrix @ vector
    
    def raise_index(self, covector: jnp.ndarray) -> jnp.ndarray:
        """
        Convert covector to vector: v^i = g^{ij} α_j
        
        This is "index raising" - the musical isomorphism ♯ (sharp).
        This is exactly what dualize() does for gradients.
        """
        return self.inverse @ covector
    
    def inner_product(self, v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute g(v1, v2) = v1^i g_{ij} v2^j
        
        The metric-induced inner product on vectors.
        """
        return v1 @ self.matrix @ v2
    
    def norm(self, v: jnp.ndarray) -> jnp.ndarray:
        """Compute ||v||_g = sqrt(g(v,v))."""
        return jnp.sqrt(self.inner_product(v, v))
    
    def transform(self, jacobian: jnp.ndarray) -> 'MetricTensor':
        """
        Transform metric under basis change: g' = J^T g J
        
        This is the covariant (0,2)-tensor transformation law.
        """
        new_matrix = jacobian.T @ self.matrix @ jacobian
        return MetricTensor(new_matrix)
    
    @classmethod
    def euclidean(cls, dim: int) -> 'MetricTensor':
        """Create identity (Euclidean) metric."""
        return cls(jnp.eye(dim))
    
    @classmethod
    def from_spd(cls, matrix: jnp.ndarray) -> 'MetricTensor':
        """Create from symmetric positive-definite matrix."""
        return cls(matrix)


@dataclass
class GeometricVector:
    """
    A vector with explicit variance and parity metadata.
    
    Used in testing and explicit geometric computations where we need
    to track the geometric type of arrays through transformations.
    """
    components: jnp.ndarray
    variance: TensorVariance
    parity: Parity = Parity.EVEN
    
    @property
    def dim(self) -> int:
        return self.components.shape[-1]
    
    def transform(self, jacobian: jnp.ndarray, 
                  det_sign: int = 1) -> 'GeometricVector':
        """
        Apply basis change transformation.
        
        Contravariant: v' = J^{-1} v
        Covariant: α' = α J
        Twisted: multiply by det_sign as well
        """
        J_inv = jnp.linalg.inv(jacobian)
        
        if self.variance == TensorVariance.CONTRAVARIANT:
            new_components = J_inv @ self.components
        elif self.variance == TensorVariance.COVARIANT:
            new_components = self.components @ jacobian
        else:
            new_components = self.components  # Scalar
        
        # Apply parity factor for twisted forms
        if self.parity == Parity.ODD:
            new_components = det_sign * new_components
        
        return GeometricVector(new_components, self.variance, self.parity)


__all__ = [
    'MetricTensor',
    'GeometricVector',
]

