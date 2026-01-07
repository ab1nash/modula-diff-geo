"""
Real-World Hypothesis Tests for Geometric Covariance

This package tests the practical advantages of geometric covariance on
simplified versions of real-world problems.

Hypotheses under test:
1. H1: SPD/Covariance Classification (EEG/BCI analog)
2. H2: Directed Graph Embedding (Social Networks/Causality)
3. H3: Chiral Discrimination (Molecular/Biological)
4. H4: Affine Invariant Recognition (Vision)
5. H5: Missing Data Reconstruction

Reference: "[2] Geometric Covariance: Math, Fields, Applications.md"
"""

from .utils import (
    DataMasker,
    MaskPattern,
    MaskedData,
    MissingDataEvaluator,
    ImputationMetrics,
    SyntheticDatasets,
)

__all__ = [
    'DataMasker',
    'MaskPattern',
    'MaskedData',
    'MissingDataEvaluator',
    'ImputationMetrics',
    'SyntheticDatasets',
]

