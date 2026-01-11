"""
Neural network components with geometric structure.

This module provides:
- GeometricModule: Base class with geometric signature tracking
- GeometricLinear: [ABC] Linear layer with explicit vector→vector signature
- FinslerLinear: Linear with asymmetric Finsler metric
- GeometricBond: Connections handling metric transitions

These extend Modula's primitives with covariant type tracking,
ensuring geometric compatibility at composition time.

See Also:
    diffgeo.geometry: Pure geometric structures
    diffgeo.optim: Geometry-aware optimization
"""

from .module import (
    GeometricModule,
    GeometricAtom,
    GeometricCompositeModule,
    GeometricCompositeAtom,
)

from .atoms import (
    GeometricLinear,
    FinslerLinear,
    TwistedEmbed,
    GeometricEmbed,
    ContactAtom,
)

from .bonds import (
    GeometricBond,
    MetricTransition,
    ParallelTransport,
    SymplecticBond,
    TransportPath,
    RopeJIT,
    create_transition_bond,
    flat_transport,
    curved_transport,
)

from .transformer import (
    GeometricAttention,
    GeometricGPT,
    StandardGPTJIT,
    TwistedEmbedWrapper,
    create_geometric_gpt,
    create_chiral_pair,
)

__all__ = [
    # Modules
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
    'RopeJIT',
    'create_transition_bond',
    'flat_transport',
    'curved_transport',
    # Transformer
    'GeometricAttention',
    'GeometricGPT',
    'StandardGPTJIT',
    'TwistedEmbedWrapper',
    'create_geometric_gpt',
    'create_chiral_pair',
]
