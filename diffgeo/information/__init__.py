"""
Information geometry: Fisher metric, divergences, and geometry extraction.

This module provides:
- FisherMetric: Fisher Information as Riemannian metric
- BregmanDivergence: Family of divergences (KL, Itakura-Saito, etc.)
- DataGeometryExtractor: Automatically learn geometry from data

Key concepts:
- Fisher Information: g_ij(θ) = E[∂_i log p · ∂_j log p]
- Natural Gradient: ∇_nat L = F^{-1} ∇L
- Bregman-Fisher duality: Hessian of Bregman generator = Fisher metric

See Also:
    diffgeo.geometry: Pure geometric structures
    diffgeo.optim: Geometry-aware optimization
"""
from .fisher import (
    FisherMetric,
    FisherAtom,
    fisher_gaussian,
    fisher_categorical,
    fisher_exponential_family,
)

from .divergence import (
    BregmanDivergence,
    KLDivergence,
    SquaredEuclidean,
    ItakuraSaito,
    LogDet,
    AlphaDivergence,
    js_divergence,
    total_variation,
    hellinger_distance,
    # Bregman-Fisher connections
    fisher_from_bregman,
    local_divergence_approximation,
    DuallyFlatManifold,
    bregman_to_statistical_manifold,
)

from .manifolds import (
    StatisticalManifold,
    empirical_fisher_from_data,
)

from .extractor import (
    DataGeometryExtractor,
    SPDGeometryExtractor,
)

__all__ = [
    # Fisher
    'FisherMetric',
    'FisherAtom',
    'fisher_gaussian',
    'fisher_categorical',
    'fisher_exponential_family',
    # Divergences
    'BregmanDivergence',
    'KLDivergence',
    'SquaredEuclidean',
    'ItakuraSaito',
    'LogDet',
    'AlphaDivergence',
    'js_divergence',
    'total_variation',
    'hellinger_distance',
    # Bregman-Fisher
    'fisher_from_bregman',
    'local_divergence_approximation',
    'DuallyFlatManifold',
    'bregman_to_statistical_manifold',
    # Manifolds
    'StatisticalManifold',
    'empirical_fisher_from_data',
    # Extractors
    'DataGeometryExtractor',
    'SPDGeometryExtractor',
]

