"""
Geometric Covariance Extensions for Modula

This package extends Modula with differential geometry primitives for
covariant pattern mining. It provides:

Core Types (diffgeo.core):
    - TensorVariance: Contravariant/covariant classification
    - Parity: Orientation behavior (even/odd)
    - MetricType: Riemannian, Finsler, Symplectic
    - GeometricSignature: Complete module type metadata

Geometry (diffgeo.geometry):
    - MetricTensor: Riemannian metric with index raising/lowering
    - GeometricVector: Vector with explicit variance and parity
    - RandersMetric: Asymmetric F(v) = sqrt(v^T A v) + b^T v
    - SPDManifold: Symmetric positive-definite matrix manifold
    - StatisticalManifold: Fisher geometry on parameter spaces

Information Geometry (diffgeo.information):
    - FisherMetric: Fisher Information as Riemannian metric
    - BregmanDivergence: Family of divergences (KL, etc.)
    - DataGeometryExtractor: Learn geometry from raw data

Neural Network (diffgeo.nn):
    - GeometricModule: Module with geometric signature tracking
    - GeometricLinear[ABC]: Linear with explicit vector→vector signature
    - FinslerLinear: Linear with asymmetric Finsler metric
    - GeometricBond: Connections handling metric transitions

Optimization (diffgeo.optim):
    - GeometricOptimizer: Base optimizer respecting manifolds
    - NaturalGradientOptimizer: Uses Fisher metric
    - FinslerOptimizer: Handles asymmetric geometry

Usage:
    from diffgeo import FinslerLinear, TwistedEmbed
    from diffgeo import TensorVariance, Parity, GeometricSignature
    from diffgeo import RandersMetric, FinslerDualizer

Reference: Design Documents in ignore-docs/
"""

# =============================================================================
# Core Types (diffgeo.core)
# =============================================================================
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

# =============================================================================
# Geometry (diffgeo.geometry)
# =============================================================================
from .geometry import (
    # Metric
    MetricTensor,
    GeometricVector,
    # Finsler
    FinslerNorm,
    RandersMetric,
    FinslerDualizer,
    finsler_orthogonalize,
    randers_geodesic_distance,
    make_randers_spd,
    # SPD
    SPDManifold,
    SPDMetricTensor,
    SPDClassifier,
    # Statistical Manifold
    StatisticalManifold,
    empirical_fisher_from_data,
)

# =============================================================================
# Information Geometry (diffgeo.information)
# =============================================================================
from .information import (
    # Fisher
    FisherMetric,
    FisherAtom,
    fisher_gaussian,
    fisher_categorical,
    fisher_exponential_family,
    # Divergences
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
    # Extractors
    DataGeometryExtractor,
    SPDGeometryExtractor,
)

# =============================================================================
# Neural Network (diffgeo.nn)
# =============================================================================
from .nn import (
    # Modules
    GeometricModule,
    GeometricAtom,
    GeometricCompositeModule,
    GeometricCompositeAtom,
    # Atoms
    GeometricLinear,
    FinslerLinear,
    TwistedEmbed,
    GeometricEmbed,
    ContactAtom,
    # Bonds
    GeometricBond,
    MetricTransition,
    ParallelTransport,
    SymplecticBond,
    TransportPath,
    RopeJIT,
    create_transition_bond,
    flat_transport,
    curved_transport,
    # Transformer
    GeometricAttention,
    GeometricGPT,
    StandardGPTJIT,
    TwistedEmbedWrapper,
    create_geometric_gpt,
    create_chiral_pair,
)

# =============================================================================
# Optimization (diffgeo.optim)
# =============================================================================
from .optim import (
    GeometricOptimizerState,
    GeometricOptimizer,
    NaturalGradientOptimizer,
    FinslerOptimizer,
    geometric_sgd_step,
    natural_gradient_step,
)


__all__ = [
    # Core enums and signatures
    "TensorVariance",
    "Parity",
    "MetricType",
    "GeometricSignature",
    "VECTOR_TO_VECTOR",
    "COVECTOR_TO_COVECTOR",
    "SCALAR_TO_VECTOR",
    "INDEX_TO_VECTOR",
    # Geometry - Metric
    "MetricTensor",
    "GeometricVector",
    # Geometry - Finsler
    "FinslerNorm",
    "RandersMetric",
    "FinslerDualizer",
    "finsler_orthogonalize",
    "randers_geodesic_distance",
    "make_randers_spd",
    # Geometry - SPD
    "SPDManifold",
    "SPDMetricTensor",
    "SPDClassifier",
    # Geometry - Statistical Manifold
    "StatisticalManifold",
    "empirical_fisher_from_data",
    # Information - Fisher
    "FisherMetric",
    "FisherAtom",
    "fisher_gaussian",
    "fisher_categorical",
    "fisher_exponential_family",
    # Information - Divergences
    "BregmanDivergence",
    "KLDivergence",
    "SquaredEuclidean",
    "ItakuraSaito",
    "LogDet",
    "AlphaDivergence",
    "js_divergence",
    "total_variation",
    "hellinger_distance",
    # Information - Bregman-Fisher
    "fisher_from_bregman",
    "local_divergence_approximation",
    "DuallyFlatManifold",
    "bregman_to_statistical_manifold",
    # Information - Extractors
    "DataGeometryExtractor",
    "SPDGeometryExtractor",
    # NN - Modules
    "GeometricModule",
    "GeometricAtom",
    "GeometricCompositeModule",
    "GeometricCompositeAtom",
    # NN - Atoms
    "GeometricLinear",
    "FinslerLinear",
    "TwistedEmbed",
    "GeometricEmbed",
    "ContactAtom",
    # NN - Bonds
    "GeometricBond",
    "MetricTransition",
    "ParallelTransport",
    "SymplecticBond",
    "TransportPath",
    "RopeJIT",
    "create_transition_bond",
    "flat_transport",
    "curved_transport",
    # NN - Transformer
    "GeometricAttention",
    "GeometricGPT",
    "StandardGPTJIT",
    "TwistedEmbedWrapper",
    "create_geometric_gpt",
    "create_chiral_pair",
    # Optimization
    "GeometricOptimizerState",
    "GeometricOptimizer",
    "NaturalGradientOptimizer",
    "FinslerOptimizer",
    "geometric_sgd_step",
    "natural_gradient_step",
]
