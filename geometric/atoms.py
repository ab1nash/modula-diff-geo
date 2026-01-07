"""
Geometric atomic primitives for the Covariant Pattern Miner.

These atoms extend Modula's trainable primitives with explicit geometric
structure. Each atom declares its geometric signature and implements
covariant operations.

Key atoms:
- GeometricLinear: Standard linear with explicit vector→vector signature
- FinslerLinear: Linear with asymmetric Finsler metric on weight space
- TwistedEmbed: Embedding sensitive to orientation (parity = -1)
- GeometricEmbed: Standard embedding with geometric tracking
- ContactAtom: Projects onto contact distribution (conservation laws)

Reference: Design Document "System Design for Pattern Miner" Sections 4.1-4.3
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Optional

from modula.atom import orthogonalize

from .core import GeometricSignature, TensorVariance, Parity, MetricType
from .module import GeometricAtom
from .finsler import RandersMetric, finsler_orthogonalize


class GeometricLinear(GeometricAtom):
    """
    Linear transformation with explicit geometric signature.
    
    This is Modula's Linear atom augmented with geometric type information.
    The signature specifies that it maps vectors to vectors (contravariant
    to contravariant), preserving the geometric character.
    
    Attributes:
        fanin: Input dimension
        fanout: Output dimension
        signature: VECTOR_TO_VECTOR (contravariant → contravariant)
    """
    
    def __init__(self, fanout: int, fanin: int,
                 parity: Parity = Parity.EVEN):
        signature = GeometricSignature(
            domain=TensorVariance.CONTRAVARIANT,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=parity,
            metric_type=MetricType.RIEMANNIAN,
            dim_in=fanin,
            dim_out=fanout
        )
        super().__init__(signature)
        
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
    
    def forward(self, inputData: jnp.ndarray, 
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """y = Wx (matrix-vector product)."""
        W = weightsList[0]
        return jnp.einsum("...ij,...j->...i", W, inputData)
    
    def initialize(self, key: jax.Array) -> List[jnp.ndarray]:
        """Initialize with orthogonalized weights."""
        W = jax.random.normal(key, shape=(self.fanout, self.fanin))
        W = orthogonalize(W) * jnp.sqrt(self.fanout / self.fanin)
        return [W]
    
    def project(self, weightsList: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Project weights back to scaled orthogonal manifold."""
        W = weightsList[0]
        W = orthogonalize(W) * jnp.sqrt(self.fanout / self.fanin)
        return [W]
    
    def dualize(self, weightGradsList: List[jnp.ndarray],
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        """Spectral dualization via Newton-Schulz orthogonalization."""
        gradW = weightGradsList[0]
        dualW = orthogonalize(gradW) * jnp.sqrt(self.fanout / self.fanin) * targetNorm
        return [dualW]


class FinslerLinear(GeometricAtom):
    """
    Linear layer with Finsler (asymmetric) metric on weight space.
    
    This atom learns both:
    - W: The linear transformation matrix
    - drift: The asymmetric "wind" direction
    
    The Finsler metric makes certain update directions "cheaper" than others,
    useful for modeling directed patterns (causal flows, time series).
    
    Reference: Design Document Section 4.1 "The FinslerLinear Atom"
    """
    
    def __init__(self, fanout: int, fanin: int,
                 drift_strength: float = 0.3,
                 fixed_drift: bool = False):
        signature = GeometricSignature(
            domain=TensorVariance.CONTRAVARIANT,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.EVEN,
            metric_type=MetricType.FINSLER,
            dim_in=fanin,
            dim_out=fanout
        )
        super().__init__(signature)
        
        self.fanin = fanin
        self.fanout = fanout
        self.drift_strength = drift_strength
        self.fixed_drift = fixed_drift
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
        self._randers_metric: Optional[RandersMetric] = None
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """Forward pass: y = Wx."""
        W = weightsList[0]
        return jnp.einsum("...ij,...j->...i", W, inputData)
    
    def initialize(self, key: jax.Array) -> List[jnp.ndarray]:
        """Initialize weights and drift vector."""
        k1, k2 = jax.random.split(key)
        
        W = jax.random.normal(k1, shape=(self.fanout, self.fanin))
        W = orthogonalize(W) * jnp.sqrt(self.fanout / self.fanin)
        
        drift = jax.random.normal(k2, shape=(self.fanout, self.fanin))
        drift = drift / jnp.linalg.norm(drift) * self.drift_strength
        
        return [W, drift]
    
    def project(self, weightsList: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Project weights to valid manifold."""
        W = weightsList[0]
        drift = weightsList[1]
        
        W = orthogonalize(W) * jnp.sqrt(self.fanout / self.fanin)
        
        drift_norm = jnp.linalg.norm(drift)
        max_drift = 0.95
        drift = jnp.where(
            drift_norm > max_drift,
            drift * (max_drift / drift_norm),
            drift
        )
        
        return [W, drift]
    
    def dualize(self, weightGradsList: List[jnp.ndarray],
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        """Finsler dualization accounting for drift."""
        gradW = weightGradsList[0]
        gradDrift = weightGradsList[1] if len(weightGradsList) > 1 else None
        
        drift = gradDrift if gradDrift is not None else jnp.zeros_like(gradW)
        
        shifted_grad = gradW - self.drift_strength * drift
        dualW = finsler_orthogonalize(shifted_grad, drift.flatten())
        dualW = dualW * jnp.sqrt(self.fanout / self.fanin) * targetNorm
        
        if self.fixed_drift or gradDrift is None:
            dualDrift = jnp.zeros_like(drift) if gradDrift is not None else drift
        else:
            dualDrift = gradDrift / (jnp.linalg.norm(gradDrift) + 1e-8)
            dualDrift = dualDrift * targetNorm * 0.1
        
        return [dualW, dualDrift]
    
    def get_asymmetry(self, weightsList: List[jnp.ndarray]) -> float:
        """Measure the learned asymmetry (drift magnitude)."""
        drift = weightsList[1]
        return float(jnp.linalg.norm(drift))


class TwistedEmbed(GeometricAtom):
    """
    Embedding layer with odd parity (orientation-sensitive).
    
    Standard embeddings are orientation-blind. TwistedEmbed tracks an
    auxiliary orientation input that flips the sign of embeddings.
    
    Use case: Chiral molecules, 3D shapes with handedness.
    
    Geometric signature: Index → TwistedVector (parity = -1)
    
    Reference: Design Document Section 4.2 "The TwistedEmbed Atom"
    """
    
    def __init__(self, dEmbed: int, numEmbed: int):
        signature = GeometricSignature(
            domain=TensorVariance.SCALAR,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.ODD,
            metric_type=MetricType.EUCLIDEAN,
            dim_in=1,
            dim_out=dEmbed
        )
        super().__init__(signature)
        
        self.numEmbed = numEmbed
        self.dEmbed = dEmbed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray],
                orientation: float = 1.0) -> jnp.ndarray:
        """Lookup embedding, scaled by orientation."""
        E = weightsList[0]
        embeddings = E[inputData]
        return embeddings * orientation
    
    def initialize(self, key: jax.Array) -> List[jnp.ndarray]:
        """Initialize with normalized embeddings."""
        E = jax.random.normal(key, shape=(self.numEmbed, self.dEmbed))
        E = E / jnp.linalg.norm(E, axis=1, keepdims=True) * jnp.sqrt(self.dEmbed)
        return [E]
    
    def project(self, weightsList: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Project embeddings to unit sphere (scaled)."""
        E = weightsList[0]
        E = E / jnp.linalg.norm(E, axis=1, keepdims=True) * jnp.sqrt(self.dEmbed)
        return [E]
    
    def dualize(self, weightGradsList: List[jnp.ndarray],
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        """Row-wise normalization for embedding gradients."""
        gradE = weightGradsList[0]
        norms = jnp.linalg.norm(gradE, axis=1, keepdims=True)
        dualE = gradE / (norms + 1e-8) * jnp.sqrt(self.dEmbed) * targetNorm
        dualE = jnp.nan_to_num(dualE)
        return [dualE]


class GeometricEmbed(GeometricAtom):
    """
    Standard embedding with explicit geometric signature (non-twisted).
    
    Maps discrete indices to continuous vectors with even parity.
    """
    
    def __init__(self, dEmbed: int, numEmbed: int):
        signature = GeometricSignature(
            domain=TensorVariance.SCALAR,
            codomain=TensorVariance.CONTRAVARIANT,
            parity=Parity.EVEN,
            metric_type=MetricType.EUCLIDEAN,
            dim_in=1,
            dim_out=dEmbed
        )
        super().__init__(signature)
        
        self.numEmbed = numEmbed
        self.dEmbed = dEmbed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """Standard embedding lookup."""
        E = weightsList[0]
        return E[inputData]
    
    def initialize(self, key: jax.Array) -> List[jnp.ndarray]:
        E = jax.random.normal(key, shape=(self.numEmbed, self.dEmbed))
        E = E / jnp.linalg.norm(E, axis=1, keepdims=True) * jnp.sqrt(self.dEmbed)
        return [E]
    
    def project(self, weightsList: List[jnp.ndarray]) -> List[jnp.ndarray]:
        E = weightsList[0]
        E = E / jnp.linalg.norm(E, axis=1, keepdims=True) * jnp.sqrt(self.dEmbed)
        return [E]
    
    def dualize(self, weightGradsList: List[jnp.ndarray],
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        gradE = weightGradsList[0]
        dualE = gradE / jnp.linalg.norm(gradE, axis=1, keepdims=True)
        dualE = dualE * jnp.sqrt(self.dEmbed) * targetNorm
        dualE = jnp.nan_to_num(dualE)
        return [dualE]


class ContactAtom(GeometricAtom):
    """
    Projects vectors onto a contact distribution (conservation constraint).
    
    A contact structure defines a hyperplane field in the tangent bundle.
    This atom learns a contact form α and projects inputs onto ker(α).
    
    Use case: Thermodynamic constraints, energy conservation.
    
    Reference: Design Document Section 4.3 "The ContactConfiguration Atom"
    """
    
    def __init__(self, dim: int):
        assert dim % 2 == 1, "Contact manifolds are odd-dimensional"
        
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
        self.smooth = True
        self.mass = 0
        self.sensitivity = 1
    
    def forward(self, inputData: jnp.ndarray,
                weightsList: List[jnp.ndarray]) -> jnp.ndarray:
        """Project input onto kernel of contact form."""
        alpha = weightsList[0]
        alpha_x = jnp.einsum("...i,i->...", inputData, alpha)
        alpha_norm_sq = jnp.dot(alpha, alpha)
        xi = alpha / (alpha_norm_sq + 1e-8)
        projection = inputData - jnp.outer(alpha_x, xi).reshape(inputData.shape)
        return projection
    
    def initialize(self, key: jax.Array) -> List[jnp.ndarray]:
        """Initialize contact form (normalized)."""
        alpha = jax.random.normal(key, shape=(self.dim,))
        alpha = alpha / jnp.linalg.norm(alpha)
        return [alpha]
    
    def project(self, weightsList: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Keep contact form normalized."""
        alpha = weightsList[0]
        alpha = alpha / jnp.linalg.norm(alpha)
        return [alpha]
    
    def dualize(self, weightGradsList: List[jnp.ndarray],
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        """Gradient on the sphere (tangent projection)."""
        gradAlpha = weightGradsList[0]
        alpha = gradAlpha
        alpha_normalized = alpha / jnp.linalg.norm(alpha)
        grad_tangent = gradAlpha - jnp.dot(gradAlpha, alpha_normalized) * alpha_normalized
        return [grad_tangent * targetNorm]


__all__ = [
    'GeometricLinear',
    'FinslerLinear',
    'TwistedEmbed',
    'GeometricEmbed',
    'ContactAtom',
]

