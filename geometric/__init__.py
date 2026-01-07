"""
Geometric Covariance Extensions for Modula

This package extends Modula with differential geometry primitives for
covariant pattern mining. It provides:

Core Types (geometric.core):
    - TensorVariance: Contravariant/covariant classification
    - Parity: Orientation behavior (even/odd)
    - MetricType: Riemannian, Finsler, Symplectic
    - GeometricSignature: Complete module type metadata

Metric Structures (geometric.metric):
    - MetricTensor: Riemannian metric with index raising/lowering
    - GeometricVector: Vector with explicit variance and parity

Finsler Geometry (geometric.finsler):
    - RandersMetric: Asymmetric F(v) = sqrt(v^T A v) + b^T v
    - FinslerDualizer: Gradient → update conversion for Finsler
    
Base Classes (geometric.module):
    - GeometricModule: Module with geometric signature tracking
    - GeometricAtom: Atomic primitive with geometric metadata

Geometric Atoms (geometric.atoms):
    - GeometricLinear: Linear with explicit vector→vector signature
    - FinslerLinear: Linear with asymmetric Finsler metric
    - TwistedEmbed: Orientation-sensitive embedding (parity=-1)
    - GeometricEmbed: Standard embedding with tracking
    - ContactAtom: Conservation law projection

Usage:
    from geometric import GeometricLinear, FinslerLinear, TwistedEmbed
    from geometric import TensorVariance, Parity, GeometricSignature
    from geometric import RandersMetric, FinslerDualizer

Reference: Design Documents in ignore-docs/
"""

# Core types
from .core import (
    TensorVariance,
    Parity,
    MetricType,
    GeometricSignature,
    VECTOR_TO_VECTOR,
    COVECTOR_TO_COVECTOR,
    SCALAR_TO_VECTOR,
    INDEX_TO_VECTOR,
)

# Metric structures
from .metric import (
    MetricTensor,
    GeometricVector,
)

# Finsler geometry
from .finsler import (
    FinslerNorm,
    RandersMetric,
    FinslerDualizer,
    finsler_orthogonalize,
    randers_geodesic_distance,
    make_randers_spd,
)

# Base classes for geometric modules
from .module import (
    GeometricModule,
    GeometricAtom,
    GeometricCompositeModule,
    GeometricCompositeAtom,
)

# Concrete geometric atoms
from .atoms import (
    GeometricLinear,
    FinslerLinear,
    TwistedEmbed,
    GeometricEmbed,
    ContactAtom,
)

# Geometric bonds
from .bonds import (
    GeometricBond,
    MetricTransition,
    ParallelTransport,
    SymplecticBond,
    TransportPath,
    create_transition_bond,
    flat_transport,
    curved_transport,
)

# Information geometry (Phase 5)
from .information import (
    FisherMetric,
    FisherAtom,
    fisher_gaussian,
    fisher_categorical,
    fisher_exponential_family,
)

# SPD manifold (Phase 5)
from .spd import (
    SPDManifold,
    SPDMetricTensor,
    SPDClassifier,
)

# Divergences (Phase 5)
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
)


__all__ = [
    # Core enums and signatures
    'TensorVariance',
    'Parity',
    'MetricType',
    'GeometricSignature',
    'VECTOR_TO_VECTOR',
    'COVECTOR_TO_COVECTOR', 
    'SCALAR_TO_VECTOR',
    'INDEX_TO_VECTOR',
    # Metric structures
    'MetricTensor',
    'GeometricVector',
    # Finsler
    'FinslerNorm',
    'RandersMetric',
    'FinslerDualizer',
    'finsler_orthogonalize',
    'randers_geodesic_distance',
    'make_randers_spd',
    # Base classes
    'GeometricModule',
    'GeometricAtom',
    'GeometricCompositeModule',
    'GeometricCompositeAtom',
    # Atoms
    'GeometricLinear',
    'FinslerLinear',
    'TwistedEmbed',
    'GeometricEmbed',
    'ContactAtom',
    # Bonds
    'GeometricBond',
    'MetricTransition',
    'ParallelTransport',
    'SymplecticBond',
    'TransportPath',
    'create_transition_bond',
    'flat_transport',
    'curved_transport',
    # Information Geometry (Phase 5)
    'FisherMetric',
    'FisherAtom',
    'fisher_gaussian',
    'fisher_categorical',
    'fisher_exponential_family',
    # SPD Manifold (Phase 5)
    'SPDManifold',
    'SPDMetricTensor',
    'SPDClassifier',
    # Divergences (Phase 5)
    'BregmanDivergence',
    'KLDivergence',
    'SquaredEuclidean',
    'ItakuraSaito',
    'LogDet',
    'AlphaDivergence',
    'js_divergence',
    'total_variation',
    'hellinger_distance',
]

