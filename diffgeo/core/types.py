"""
Core geometric type system for covariant pattern mining.

This module defines the fundamental type taxonomy from Burke's
"Applied Differential Geometry":

- TensorVariance: How tensors transform under basis change
- Parity: Orientation behavior under reflection  
- MetricType: Classification of metric structures
- GeometricSignature: Complete module type metadata

The key insight is that vectors and covectors (gradients) are geometrically
distinct objects. "Metric blindness" - treating them identically - leads to
coordinate-dependent artifacts.

Reference: Burke "Applied Differential Geometry", Chapter on Descriptive Geometry
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TensorVariance(Enum):
    """
    Classification of tensor transformation behavior.
    
    Under a basis change with Jacobian J:
    - CONTRAVARIANT: v' = J^{-1}v  (velocities, displacements)
    - COVARIANT: α' = αJ         (gradients, 1-forms)
    - MIXED: Has both upper and lower indices (operators)
    - SCALAR: Invariant under all transformations
    
    The fundamental operation v^i α_i (contraction) yields a scalar precisely
    because the transformation factors cancel: (J^{-1}v)^T (αJ) = v^T α.
    """
    CONTRAVARIANT = "contra"   # Upper indices, transforms with J^{-1}
    COVARIANT = "co"           # Lower indices, transforms with J
    MIXED = "mixed"            # Both upper and lower indices
    SCALAR = "scalar"          # Rank 0, invariant


class Parity(Enum):
    """
    Orientation behavior under reflection (parity transformation).
    
    - EVEN (+1): True tensors, invariant under reflection
    - ODD (-1): Pseudotensors (twisted forms), flip sign under reflection
    
    Physical examples:
    - Electric field E: EVEN (polar vector)
    - Magnetic field B: ODD (axial/pseudo vector)
    
    Parity composition: EVEN × EVEN = EVEN, EVEN × ODD = ODD, ODD × ODD = EVEN
    This is isomorphic to Z_2 group multiplication.
    """
    EVEN = 1    # True tensor: no sign flip under reflection
    ODD = -1    # Pseudotensor: flips sign under det(J) = -1
    
    def __mul__(self, other: 'Parity') -> 'Parity':
        """Compose parities: Z_2 multiplication."""
        return Parity(self.value * other.value)
    
    @classmethod
    def from_int(cls, value: int) -> 'Parity':
        """Convert ±1 to Parity enum."""
        return cls.EVEN if value == 1 else cls.ODD


class MetricType(Enum):
    """
    Classification of metric structures on the weight space.
    
    - RIEMANNIAN: Symmetric, quadratic metric (g_ij v^i v^j)
    - FINSLER: General Minkowski norm, potentially asymmetric
    - SYMPLECTIC: Anti-symmetric pairing (Hamiltonian systems)
    - EUCLIDEAN: Special case of Riemannian with identity metric
    """
    EUCLIDEAN = "euclidean"     # Flat, identity metric
    RIEMANNIAN = "riemannian"   # Symmetric positive-definite
    FINSLER = "finsler"         # Non-quadratic norm (asymmetric)
    SYMPLECTIC = "symplectic"   # Anti-symmetric 2-form


@dataclass(frozen=True)
class GeometricSignature:
    """
    Complete geometric metadata defining a module's input/output types.
    
    This is the "type signature" in the geometric sense - it specifies:
    1. What kind of tensor the module accepts (domain)
    2. What kind of tensor it produces (codomain)
    3. Whether it preserves or flips orientation (parity)
    4. What metric structure it respects (metric_type)
    
    The @ (composition) operator checks compatibility:
    - codomain of inner must match domain of outer
    - parities multiply (Z_2 composition)
    - metric types must be compatible or have explicit transition
    
    Attributes:
        domain: TensorVariance of input
        codomain: TensorVariance of output
        parity: Orientation behavior (even=true tensor, odd=pseudo)
        metric_type: What metric structure the module respects
        dim_in: Input dimension (optional, for shape checking)
        dim_out: Output dimension (optional)
    """
    domain: TensorVariance
    codomain: TensorVariance
    parity: Parity = Parity.EVEN
    metric_type: MetricType = MetricType.RIEMANNIAN
    dim_in: Optional[int] = None
    dim_out: Optional[int] = None
    
    def is_compatible_with(self, other: 'GeometricSignature') -> bool:
        """
        Check if this module's output can feed into other's input.
        
        Compatibility requires codomain = other.domain (type matching).
        Dimension matching is checked only if both specify dimensions.
        """
        if self.codomain != other.domain:
            # Could potentially auto-convert via metric, but strict for now
            if not (self.codomain == TensorVariance.SCALAR or 
                    other.domain == TensorVariance.SCALAR):
                return False
        
        # Check dimension compatibility if specified
        if self.dim_out is not None and other.dim_in is not None:
            if self.dim_out != other.dim_in:
                return False
        
        return True
    
    def compose_with(self, other: 'GeometricSignature') -> 'GeometricSignature':
        """
        Compute signature of composition (self @ other means other then self).
        
        Input comes from other.domain, output goes to self.codomain.
        Parity multiplies (Z_2 composition).
        """
        if not other.is_compatible_with(self):
            raise TypeError(
                f"Geometric type mismatch: {other.codomain} -> {self.domain}"
            )
        
        return GeometricSignature(
            domain=other.domain,
            codomain=self.codomain,
            parity=self.parity * other.parity,
            metric_type=self.metric_type,  # Outer determines metric
            dim_in=other.dim_in,
            dim_out=self.dim_out
        )


# =============================================================================
# Common Signatures
# =============================================================================

VECTOR_TO_VECTOR = GeometricSignature(
    TensorVariance.CONTRAVARIANT, 
    TensorVariance.CONTRAVARIANT
)

COVECTOR_TO_COVECTOR = GeometricSignature(
    TensorVariance.COVARIANT,
    TensorVariance.COVARIANT  
)

SCALAR_TO_VECTOR = GeometricSignature(
    TensorVariance.SCALAR,
    TensorVariance.CONTRAVARIANT
)

INDEX_TO_VECTOR = GeometricSignature(
    TensorVariance.SCALAR,  # Index is scalar-like
    TensorVariance.CONTRAVARIANT
)


__all__ = [
    'TensorVariance',
    'Parity',
    'MetricType',
    'GeometricSignature',
    'VECTOR_TO_VECTOR',
    'COVECTOR_TO_COVECTOR',
    'SCALAR_TO_VECTOR',
    'INDEX_TO_VECTOR',
]

