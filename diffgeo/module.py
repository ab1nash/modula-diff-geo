"""
Base classes for geometric modules extending Modula.

This module provides:
- GeometricModule: Base class with geometric signature tracking
- GeometricAtom: Atomic primitive with geometric metadata
- GeometricCompositeModule: Composite with signature propagation
- GeometricCompositeAtom: Composite atoms with signature tracking

These classes extend Modula's Module/Atom with covariant type tracking,
ensuring geometric compatibility is checked at composition time.

Reference: Design Document "System Design for Pattern Miner"
"""
from __future__ import annotations

import jax.numpy as jnp
from typing import Any, Optional, Union, List

from modula.abstract import Module, Atom, CompositeModule

from .core import GeometricSignature, TensorVariance, Parity
from .metric import MetricTensor


class GeometricModule(Module):
    """
    Abstract base class for modules with geometric covariance tracking.
    
    Extends Modula's Module with:
    - Explicit geometric signature (domain/codomain types)
    - Parity (orientation) tracking through composition
    - Metric-aware dualization
    
    The key methods extended:
    - __matmul__: Checks geometric compatibility before composition
    - dualize: Uses metric structure for proper gradient→update conversion
    
    Pattern Theory mapping:
    - GeometricModule = Generator (g ∈ G) with geometric variance
    - Composition = Connector (σ) that validates type compatibility
    """
    
    def __init__(self, signature: GeometricSignature):
        super().__init__()
        self._signature = signature
        self._metric: Optional[MetricTensor] = None
    
    @property
    def signature(self) -> GeometricSignature:
        """Get the geometric signature of this module."""
        return self._signature
    
    @property
    def is_twisted(self) -> bool:
        """Check if this module has odd parity (pseudotensor behavior)."""
        return self._signature.parity == Parity.ODD
    
    @property
    def metric(self) -> Optional[MetricTensor]:
        """Get the metric tensor if one is defined."""
        return self._metric
    
    def set_metric(self, metric: MetricTensor) -> None:
        """Set the metric tensor for this module."""
        self._metric = metric
    
    def check_compatibility(self, other: 'GeometricModule') -> bool:
        """Check if composition self @ other is geometrically valid."""
        return other.signature.is_compatible_with(self.signature)
    
    def __matmul__(self, other_module: Union[Module, 'GeometricModule', tuple]) -> Module:
        """Override composition to check geometric compatibility."""
        from modula.abstract import TupleModule
        
        if isinstance(other_module, tuple):
            other_module = TupleModule(other_module)
        
        if isinstance(other_module, GeometricModule):
            if not self.check_compatibility(other_module):
                raise TypeError(
                    f"Geometric type mismatch in composition:\n"
                    f"  Inner module produces: {other_module.signature.codomain}\n"
                    f"  Outer module expects: {self.signature.domain}"
                )
            return GeometricCompositeModule(self, other_module)
        
        return CompositeModule(self, other_module)
    
    def dualize(self, weightGradsList: List[jnp.ndarray], 
                targetNorm: float = 1.0) -> List[jnp.ndarray]:
        """
        Convert gradient (covector) to update direction (vector).
        
        The gradient ∇L lives in the cotangent space. To update weights,
        we need a vector. The metric provides the conversion:
            Δw = g^{-1}(∇L)
        """
        return self._dualize_impl(weightGradsList, targetNorm)
    
    def _dualize_impl(self, weightGradsList: List[jnp.ndarray],
                      targetNorm: float) -> List[jnp.ndarray]:
        """Default dualization: apply metric if available."""
        if self._metric is not None:
            return [self._metric.raise_index(g) * targetNorm 
                    for g in weightGradsList]
        else:
            return [g * targetNorm for g in weightGradsList]


class GeometricAtom(Atom):
    """
    Base class for atomic (single-weight) geometric modules.
    
    Combines Modula's Atom (trainable primitive) with geometric type tracking.
    Uses composition rather than multiple inheritance to avoid MRO issues.
    
    Concrete implementations include:
    - GeometricLinear: Standard linear with explicit signature
    - FinslerLinear: Linear with Finsler (asymmetric) metric
    - TwistedEmbed: Embedding that tracks orientation
    """
    
    def __init__(self, signature: GeometricSignature):
        super().__init__()
        self._signature = signature
        self._metric: Optional[MetricTensor] = None
    
    @property
    def signature(self) -> GeometricSignature:
        """Get the geometric signature of this module."""
        return self._signature
    
    @property
    def is_twisted(self) -> bool:
        """Check if this module has odd parity."""
        return self._signature.parity == Parity.ODD
    
    @property
    def metric(self) -> Optional[MetricTensor]:
        """Get the metric tensor if one is defined."""
        return self._metric
    
    def set_metric(self, metric: MetricTensor) -> None:
        """Set the metric tensor for this module."""
        self._metric = metric
    
    def check_compatibility(self, other: 'GeometricAtom') -> bool:
        """Check if composition self @ other is geometrically valid."""
        if hasattr(other, 'signature'):
            return other.signature.is_compatible_with(self.signature)
        return True
    
    def __matmul__(self, other_module):
        """Override composition to check geometric compatibility."""
        from modula.abstract import TupleModule
        
        if isinstance(other_module, tuple):
            other_module = TupleModule(other_module)
        
        # Handle both GeometricAtom and GeometricCompositeAtom
        if hasattr(other_module, 'signature'):
            if not self.check_compatibility(other_module):
                raise TypeError(
                    f"Geometric type mismatch in composition:\n"
                    f"  Inner module produces: {other_module.signature.codomain}\n"
                    f"  Outer module expects: {self.signature.domain}"
                )
            return GeometricCompositeAtom(self, other_module)
        
        return CompositeModule(self, other_module)


class GeometricCompositeModule(CompositeModule):
    """
    Composite module that tracks geometric signature through composition.
    """
    
    def __init__(self, outer_module: GeometricModule, inner_module: GeometricModule):
        super().__init__(outer_module, inner_module)
        composed_sig = outer_module.signature.compose_with(inner_module.signature)
        self._signature = composed_sig
    
    @property
    def signature(self) -> GeometricSignature:
        return self._signature
    
    @property
    def is_twisted(self) -> bool:
        return self._signature.parity == Parity.ODD


class GeometricCompositeAtom(CompositeModule):
    """
    Composite of geometric atoms that tracks signature through composition.
    """
    
    def __init__(self, outer_atom, inner_atom):
        """
        Args:
            outer_atom: Module with signature (GeometricAtom or GeometricCompositeAtom)
            inner_atom: Module with signature (GeometricAtom or GeometricCompositeAtom)
        """
        super().__init__(outer_atom, inner_atom)
        composed_sig = outer_atom.signature.compose_with(inner_atom.signature)
        self._signature = composed_sig
    
    @property
    def signature(self) -> GeometricSignature:
        return self._signature
    
    @property
    def is_twisted(self) -> bool:
        return self._signature.parity == Parity.ODD
    
    def check_compatibility(self, other) -> bool:
        """Check composition compatibility."""
        if hasattr(other, 'signature'):
            return other.signature.is_compatible_with(self.signature)
        return True
    
    def __matmul__(self, other_module):
        """Allow further composition with geometric modules."""
        from modula.abstract import TupleModule
        
        if isinstance(other_module, tuple):
            other_module = TupleModule(other_module)
        
        # If other has signature, preserve geometric composition
        if hasattr(other_module, 'signature'):
            if not self.check_compatibility(other_module):
                raise TypeError(
                    f"Geometric type mismatch in composition:\n"
                    f"  Inner module produces: {other_module.signature.codomain}\n"
                    f"  Outer module expects: {self.signature.domain}"
                )
            return GeometricCompositeAtom(self, other_module)
        
        return CompositeModule(self, other_module)


__all__ = [
    'GeometricModule',
    'GeometricAtom',
    'GeometricCompositeModule',
    'GeometricCompositeAtom',
]

