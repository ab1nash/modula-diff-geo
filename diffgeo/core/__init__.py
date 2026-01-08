"""
Core type system for geometric covariance.

This module provides the fundamental type taxonomy:
- TensorVariance: How tensors transform (contravariant/covariant)
- Parity: Orientation behavior (even/odd)
- MetricType: Classification of metric structures
- GeometricSignature: Complete module type metadata
"""
from .types import (
    TensorVariance,
    Parity,
    MetricType,
    GeometricSignature,
    VECTOR_TO_VECTOR,
    COVECTOR_TO_COVECTOR,
    SCALAR_TO_VECTOR,
    INDEX_TO_VECTOR,
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

