"""
Pure geometry module: metrics, manifolds, and geometric structures.

This module provides:
- MetricTensor: Riemannian metric for index operations
- GeometricVector: Vector with explicit variance/parity
- RandersMetric: Finsler geometry for asymmetric distances
- SPDManifold: Symmetric positive-definite matrix manifold
- StatisticalManifold: Fisher geometry on parameter spaces

See Also:
    diffgeo.information: Fisher information and divergences
    diffgeo.nn: Neural network layers with geometric structure
"""
from .metric import (
    MetricTensor,
    GeometricVector,
)

from .finsler import (
    FinslerNorm,
    RandersMetric,
    FinslerDualizer,
    finsler_orthogonalize,
    randers_geodesic_distance,
    make_randers_spd,
)

from .spd import (
    SPDManifold,
    SPDMetricTensor,
    SPDClassifier,
)

# StatisticalManifold is in diffgeo.information (due to Fisher dependency)
# Re-exported here for convenience
from ..information.manifolds import (
    StatisticalManifold,
    empirical_fisher_from_data,
)

__all__ = [
    # Metric
    'MetricTensor',
    'GeometricVector',
    # Finsler
    'FinslerNorm',
    'RandersMetric',
    'FinslerDualizer',
    'finsler_orthogonalize',
    'randers_geodesic_distance',
    'make_randers_spd',
    # SPD
    'SPDManifold',
    'SPDMetricTensor',
    'SPDClassifier',
    # Statistical Manifold
    'StatisticalManifold',
    'empirical_fisher_from_data',
]

