"""
Geometry-aware optimization.

This module provides:
- GeometricOptimizer: Base optimizer respecting manifold structure
- NaturalGradientOptimizer: Uses Fisher metric for natural gradient
- FinslerOptimizer: Handles asymmetric Finsler geometry

Key insight: Gradients are covectors, updates are vectors.
The metric converts between them: Δw = g^{-1}(∇L)

See Also:
    diffgeo.geometry: Metric structures
    diffgeo.information: Fisher metric computation
"""
from .optimizer import (
    GeometricOptimizerState,
    GeometricOptimizer,
    NaturalGradientOptimizer,
    FinslerOptimizer,
    geometric_sgd_step,
    natural_gradient_step,
)

__all__ = [
    'GeometricOptimizerState',
    'GeometricOptimizer',
    'NaturalGradientOptimizer',
    'FinslerOptimizer',
    'geometric_sgd_step',
    'natural_gradient_step',
]

