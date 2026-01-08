"""
Geometric bonds for connecting modules with different metric structures.

This module provides:
- MetricTransition: Bond handling metric mismatches between layers
- ParallelTransport: Transport vectors between tangent spaces

These bonds enable composition of modules with different geometric structures
while maintaining mathematical consistency.

Reference: Design Document Section 5 "Connectors and Configurations"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Optional, Tuple
from dataclasses import dataclass

from modula.abstract import Bond

from ..core import GeometricSignature, TensorVariance, Parity, MetricType
from ..geometry.metric import MetricTensor


class GeometricBond(Bond):
    """
    Base class for geometric bonds.
    
    Geometric bonds are parameter-free connections between modules that
    may perform geometric operations like metric transitions or parallel
    transport.
    """
    
    def __init__(self, signature: GeometricSignature):
        super().__init__()
        self._signature = signature
    
    @property
    def signature(self) -> GeometricSignature:
        return self._signature
    
    @property
    def is_twisted(self) -> bool:
        return self._signature.parity == Parity.ODD


class MetricTransition(GeometricBond):
    """
    Bond that handles transition between different metric structures.
    
    When composing modules with mismatched metrics (e.g., Riemannian to
    Finsler), this bond provides a smooth interpolation that accounts
    for the energy cost of the transition.
    
    The transition can model:
    - Going from isotropic to anisotropic space
    - Adding/removing drift (Finsler asymmetry)
    - Changing the local geometry
    
    Mathematical properties:
    - Identity when source and target metrics match (T4.3.3)
    - Smooth interpolation (T4.3.2)
    - Directional cost asymmetry (T4.3.1)
    """
    
    def __init__(self, 
                 source_metric: MetricType,
                 target_metric: MetricType,
                 dim: int,
                 transition_strength: float = 1.0):
        signature = GeometricSignature(
            domain=TensorVariance.CONTRAVARIANT,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.EVEN,
            metric_type=target_metric,
            dim_in=dim,
            dim_out=dim
        )
        super().__init__(signature)
        
        self.source_metric = source_metric
        self.target_metric = target_metric
        self.dim = dim
        self.transition_strength = transition_strength
        self.smooth = True
        self.sensitivity = 1.0
    
    def forward(self, inputData: jnp.ndarray, 
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Apply metric transition.
        
        For now, this is identity (the transition affects dualization,
        not forward pass). In full implementation, could apply a learned
        or fixed transition map.
        """
        return inputData
    
    def compute_transition_cost(self, 
                                 v: jnp.ndarray,
                                 source_metric: Optional[MetricTensor] = None,
                                 target_metric: Optional[MetricTensor] = None) -> float:
        """
        Compute the cost of transitioning vector v between metrics.
        
        The cost is higher when transitioning "against" the natural
        geometry (e.g., against drift in Finsler).
        """
        if source_metric is None or target_metric is None:
            return 0.0
        
        # Cost is the difference in norms
        source_norm = source_metric.norm(v)
        target_norm = target_metric.norm(v)
        
        return float(jnp.abs(source_norm - target_norm))
    
    def is_identity(self) -> bool:
        """Check if transition is identity (same metrics)."""
        return self.source_metric == self.target_metric


class ParallelTransport(GeometricBond):
    """
    Parallel transport of vectors along a geodesic.
    
    In Riemannian geometry, parallel transport moves a vector from one
    tangent space to another while keeping it "as parallel as possible"
    (minimizing covariant derivative along the path).
    
    Properties:
    - Length preservation (T4.2.1): ||Γ(v)|| = ||v||
    - Angle preservation (T4.2.2): Inner products preserved
    - Path independence on flat manifolds (T4.2.3)
    - Holonomy on curved manifolds (T4.2.4)
    
    For the weight space of neural networks:
    - The manifold is the space of weight matrices
    - Parallel transport ensures gradients are "aligned" across layers
    - On flat (Euclidean) manifolds, this is identity
    """
    
    def __init__(self,
                 dim: int,
                 connection_type: str = "levi_civita",
                 curvature: float = 0.0):
        """
        Args:
            dim: Dimension of vectors being transported
            connection_type: Type of connection ("levi_civita", "flat", "custom")
            curvature: Scalar curvature (0 for flat space)
        """
        signature = GeometricSignature(
            domain=TensorVariance.CONTRAVARIANT,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.EVEN,
            metric_type=MetricType.RIEMANNIAN,
            dim_in=dim,
            dim_out=dim
        )
        super().__init__(signature)
        
        self.dim = dim
        self.connection_type = connection_type
        self.curvature = curvature
        self.smooth = True
        self.sensitivity = 1.0
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Apply parallel transport.
        
        For flat connections, this is identity.
        For curved connections, applies the appropriate rotation.
        """
        if self.connection_type == "flat" or self.curvature == 0.0:
            return inputData
        
        # For non-flat, apply holonomy correction
        # This is a simplified model - full implementation would use
        # the connection coefficients (Christoffel symbols)
        return self._transport_with_curvature(inputData)
    
    def _transport_with_curvature(self, v: jnp.ndarray) -> jnp.ndarray:
        """
        Transport with curvature correction.
        
        For small curvature, the holonomy is approximately:
        Γ(v) ≈ v + κ * (rotation in plane of transport)
        
        This is a simplified model for testing.
        """
        # For demonstration, apply small rotation proportional to curvature
        # In practice, this would depend on the path and full connection
        if len(v.shape) == 1 and self.dim >= 2:
            # Apply small rotation in first two dimensions
            angle = self.curvature * 0.1
            c, s = jnp.cos(angle), jnp.sin(angle)
            v_new = v.at[0].set(c * v[0] - s * v[1])
            v_new = v_new.at[1].set(s * v[0] + c * v[1])
            return v_new
        return v
    
    def compute_holonomy(self, loop_area: float) -> jnp.ndarray:
        """
        Compute holonomy around a closed loop.
        
        For a loop enclosing area A on a surface with Gaussian curvature K,
        the holonomy angle is approximately K * A (Gauss-Bonnet).
        
        Returns rotation matrix representing the holonomy.
        """
        angle = self.curvature * loop_area
        c, s = jnp.cos(angle), jnp.sin(angle)
        
        # Return identity with rotation in first 2x2 block
        holonomy = jnp.eye(self.dim)
        if self.dim >= 2:
            holonomy = holonomy.at[0, 0].set(c)
            holonomy = holonomy.at[0, 1].set(-s)
            holonomy = holonomy.at[1, 0].set(s)
            holonomy = holonomy.at[1, 1].set(c)
        
        return holonomy
    
    def preserves_length(self, v: jnp.ndarray, 
                         metric: Optional[MetricTensor] = None) -> bool:
        """
        Verify that transport preserves vector length.
        
        For Levi-Civita connection, this should always be true.
        """
        transported = self.forward(v, [])
        
        if metric is None:
            # Euclidean norm
            return jnp.allclose(jnp.linalg.norm(v), jnp.linalg.norm(transported))
        else:
            return jnp.allclose(metric.norm(v), metric.norm(transported))


class SymplecticBond(GeometricBond):
    """
    Bond that preserves symplectic structure.
    
    Used for Hamiltonian dynamics where we need to preserve:
    - The symplectic 2-form ω
    - Phase space volume (Liouville's theorem)
    - Energy along trajectories
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension (must be even for symplectic structure)
        """
        assert dim % 2 == 0, "Symplectic manifolds are even-dimensional"
        
        signature = GeometricSignature(
            domain=TensorVariance.CONTRAVARIANT,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.EVEN,
            metric_type=MetricType.SYMPLECTIC,
            dim_in=dim,
            dim_out=dim
        )
        super().__init__(signature)
        
        self.dim = dim
        self.n = dim // 2  # Number of (q, p) pairs
        self.smooth = True
        self.sensitivity = 1.0
        
        # Standard symplectic matrix J = [[0, I], [-I, 0]]
        self._J = self._build_symplectic_matrix()
    
    def _build_symplectic_matrix(self) -> jnp.ndarray:
        """Build standard symplectic matrix."""
        n = self.n
        J = jnp.zeros((self.dim, self.dim))
        J = J.at[:n, n:].set(jnp.eye(n))
        J = J.at[n:, :n].set(-jnp.eye(n))
        return J
    
    @property
    def J(self) -> jnp.ndarray:
        """The symplectic matrix."""
        return self._J
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """Apply symplectic transformation (identity for base bond)."""
        return inputData
    
    def hamiltonian_vector_field(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Convert gradient to Hamiltonian vector field: X_H = J ∇H
        
        This is the symplectic analog of gradient descent.
        In Hamiltonian mechanics, motion follows J∇H, not -∇H.
        """
        return self.J @ gradient
    
    def preserves_symplectic_form(self, matrix: jnp.ndarray) -> bool:
        """
        Check if a linear map preserves the symplectic form.
        
        A matrix M is symplectic iff M^T J M = J.
        """
        result = matrix.T @ self.J @ matrix
        return jnp.allclose(result, self.J, rtol=1e-5)


@dataclass
class TransportPath:
    """
    Represents a path for parallel transport.
    
    The path can be:
    - Linear (geodesic in flat space)
    - Curved (specified by intermediate points or tangent vectors)
    """
    start_point: jnp.ndarray
    end_point: jnp.ndarray
    intermediate_points: Optional[jnp.ndarray] = None
    
    @property
    def is_geodesic(self) -> bool:
        """Check if path is a geodesic (straight line in flat space)."""
        return self.intermediate_points is None
    
    @property
    def path_length(self) -> float:
        """Compute the total path length."""
        if self.is_geodesic:
            return float(jnp.linalg.norm(self.end_point - self.start_point))
        
        # Sum segment lengths
        points = jnp.vstack([
            self.start_point[None, :],
            self.intermediate_points,
            self.end_point[None, :]
        ])
        segments = jnp.diff(points, axis=0)
        return float(jnp.sum(jnp.linalg.norm(segments, axis=1)))


# =============================================================================
# Utility Functions
# =============================================================================

def create_transition_bond(source: MetricType, 
                           target: MetricType,
                           dim: int) -> MetricTransition:
    """Factory function for creating metric transition bonds."""
    return MetricTransition(source, target, dim)


def flat_transport(dim: int) -> ParallelTransport:
    """Create flat (Euclidean) parallel transport."""
    return ParallelTransport(dim, connection_type="flat", curvature=0.0)


def curved_transport(dim: int, curvature: float) -> ParallelTransport:
    """Create parallel transport with specified curvature."""
    return ParallelTransport(dim, connection_type="levi_civita", curvature=curvature)


__all__ = [
    'GeometricBond',
    'MetricTransition',
    'ParallelTransport',
    'SymplecticBond',
    'TransportPath',
    'create_transition_bond',
    'flat_transport',
    'curved_transport',
]

