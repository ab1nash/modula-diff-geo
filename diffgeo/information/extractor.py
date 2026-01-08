"""
Data Geometry Extractor: Automatic Manifold Discovery from Raw Data

This module provides tools to automatically extract geometric structure
from raw observations, without requiring hand-crafted manifold definitions.

=============================================================================
THE CORE PROBLEM (from research doc [3]):
=============================================================================

Currently, we use pre-computed covariance matrices for EEG data. But:
  - Not all datasets have natural covariance structure
  - We want geometry to EMERGE from data, not be hand-crafted
  - Fisher Information is UNIVERSAL - works for any parametric family

This module bridges raw data → StatisticalManifold → Fisher metric.

=============================================================================
KEY INSIGHT:
=============================================================================

ANY dataset defines a statistical manifold if we can fit a probabilistic
model to it. The geometry then emerges from the Fisher Information of
that model.

For example:
  - Time series → fit temporal model → Fisher captures correlation structure
  - Images → fit spatial model → Fisher captures texture patterns
  - Graphs → fit edge distribution → Fisher captures connectivity

=============================================================================
CONNECTIONS:
=============================================================================

  ┌─────────────────────────────────────────────────────────────────────┐
  │  RAW DATA (time series, images, graphs, etc.)                       │
  │    ↓ extract                                                        │
  │  geometry_extractor.py  (THIS FILE)                                 │
  │    ↓ creates                                                        │
  │  statistical_manifold.py  →  StatisticalManifold with Fisher metric │
  │    ↓ used by                                                        │
  │  optimizer.py  →  Geometric gradient descent                        │
  └─────────────────────────────────────────────────────────────────────┘

See also:
  - spd.py: SPD manifold for covariance matrices
  - information.py: Fisher metric computation
  - core.py: TensorVariance for covariant/contravariant classification
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Literal
from functools import partial

# ─────────────────────────────────────────────────────────────────────────────
# Internal imports
# ─────────────────────────────────────────────────────────────────────────────
from ..core import TensorVariance, MetricType
from ..geometry.metric import MetricTensor
from .fisher import FisherMetric
from ..geometry.finsler import RandersMetric
from .manifolds import StatisticalManifold


# =============================================================================
# DATA GEOMETRY EXTRACTOR: Main Interface
# =============================================================================

class DataGeometryExtractor:
    """
    Extract geometric structure (Fisher manifold) from raw data.
    
    This is the BRIDGE between raw observations and geometric analysis.
    Instead of requiring users to specify manifold structure, we LEARN it.
    
    ==========================================================================
    EXTRACTION MODES:
    ==========================================================================
    
    1. TIME SERIES: Extract temporal covariance structure
       → Fits multivariate Gaussian, Fisher metric = inverse covariance
       
    2. DISTRIBUTION: Fit parametric distribution to data
       → General Fisher from score function
       
    3. NEURAL NETWORK: Use model's output distribution
       → Empirical Fisher from per-sample gradients
       
    4. PAIRWISE DISTANCES: Infer manifold from distance matrix
       → MDS-style embedding, then local Fisher
    
    ==========================================================================
    USAGE:
    ==========================================================================
    
        # From time series
        extractor = DataGeometryExtractor()
        manifold = extractor.from_time_series(eeg_signals, window_size=64)
        
        # From neural network predictions
        manifold = extractor.from_neural_network(model, params, data)
        
        # Use for geometric optimization
        natural_grad = manifold.natural_gradient(euclidean_grad)
    
    ==========================================================================
    CONNECTION TO SPD MANIFOLD:
    ==========================================================================
    
    For time series, the extracted geometry matches the SPD manifold of
    covariance matrices (see spd.py). The Fisher metric on Gaussian
    parameters gives the same geometry as the affine-invariant metric
    on SPD(n). This validates our approach!
    """
    
    def __init__(self, 
                 regularization: float = 1e-6,
                 detect_asymmetry: bool = True):
        """
        Initialize extractor.
        
        Args:
            regularization: Diagonal regularization for numerical stability
            detect_asymmetry: Whether to check for Finsler structure
        """
        self.regularization = regularization
        self.detect_asymmetry = detect_asymmetry
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACTION METHOD 1: Time Series → Covariance Manifold
    # ─────────────────────────────────────────────────────────────────────────
    
    def from_time_series(self,
                         data: jnp.ndarray,
                         window_size: Optional[int] = None,
                         overlap: float = 0.5) -> StatisticalManifold:
        """
        Extract SPD covariance geometry from multivariate time series.
        
        This is what we currently do for EEG, but now automated:
        1. Compute sample covariance from time windows
        2. Model as multivariate Gaussian
        3. Fisher metric = inverse covariance (precision)
        
        For Gaussian N(μ, Σ) with known Σ, Fisher metric on μ is Σ^{-1}.
        This matches the Riemannian metric on SPD manifold!
        
        Args:
            data: Time series (n_timepoints, n_channels) or
                  (n_samples, n_timepoints, n_channels)
            window_size: Size of time windows (default: n_timepoints)
            overlap: Fraction of overlap between windows
            
        Returns:
            StatisticalManifold with covariance-based Fisher metric
            
        See also:
            - spd.py: SPDManifold for explicit covariance matrix geometry
            - Research doc [2] Section 4: "The Riemannian Geometry of SPD Matrices"
        """
        data = jnp.asarray(data)
        
        # Handle different input shapes
        if data.ndim == 2:
            # Single time series: (n_timepoints, n_channels)
            n_timepoints, n_channels = data.shape
            data = data[None, :, :]  # Add batch dimension
        elif data.ndim == 3:
            # Multiple samples: (n_samples, n_timepoints, n_channels)
            _, n_timepoints, n_channels = data.shape
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
        
        # Default window size
        if window_size is None:
            window_size = n_timepoints
        
        # Compute covariance matrices from windows
        covariances = self._compute_windowed_covariances(
            data, window_size, overlap
        )
        
        # Mean covariance (Fréchet mean would be better, but arithmetic is faster)
        mean_cov = jnp.mean(covariances, axis=0)
        
        # Regularize for numerical stability
        mean_cov = mean_cov + self.regularization * jnp.eye(n_channels)
        
        # Mean of the data (flatten across windows)
        mean_data = jnp.mean(data.reshape(-1, n_channels), axis=0)
        
        # Create Gaussian manifold
        # Fisher metric for Gaussian mean with known covariance is Σ^{-1}
        return StatisticalManifold.from_gaussian(
            mean=mean_data,
            covariance=mean_cov,
            samples=data.reshape(-1, n_channels)
        )
    
    def _compute_windowed_covariances(self,
                                       data: jnp.ndarray,
                                       window_size: int,
                                       overlap: float) -> jnp.ndarray:
        """
        Compute sample covariances from sliding windows.
        
        Args:
            data: (n_samples, n_timepoints, n_channels)
            window_size: Window length in timepoints
            overlap: Fraction of overlap
            
        Returns:
            Covariance matrices (n_windows, n_channels, n_channels)
        """
        n_samples, n_timepoints, n_channels = data.shape
        step = int(window_size * (1 - overlap))
        step = max(1, step)  # At least 1
        
        covariances = []
        
        for sample in range(n_samples):
            for start in range(0, n_timepoints - window_size + 1, step):
                window = data[sample, start:start + window_size, :]
                # Sample covariance
                centered = window - jnp.mean(window, axis=0, keepdims=True)
                cov = (centered.T @ centered) / (window_size - 1)
                covariances.append(cov)
        
        return jnp.stack(covariances)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACTION METHOD 2: Neural Network → Empirical Fisher
    # ─────────────────────────────────────────────────────────────────────────
    
    def from_neural_network(self,
                            model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            params: jnp.ndarray,
                            inputs: jnp.ndarray,
                            targets: jnp.ndarray,
                            loss_type: Literal['mse', 'cross_entropy'] = 'mse'
                            ) -> StatisticalManifold:
        """
        Extract Fisher geometry from neural network predictions.
        
        The empirical Fisher uses the model's output distribution:
            F = E_{x~data}[∇log p(y|x,θ) ∇log p(y|x,θ)^T]
        
        This captures:
          - Which parameters are "stiff" (well-constrained by data)
          - Which are "sloppy" (can vary without affecting loss)
          - The effective dimensionality of the model
        
        Args:
            model_fn: Function (params, x) -> predictions
            params: Flattened model parameters
            inputs: Input data batch
            targets: Target values
            loss_type: 'mse' for regression, 'cross_entropy' for classification
            
        Returns:
            StatisticalManifold with empirical Fisher metric
            
        See also:
            - information.py: FisherAtom.empirical_fisher()
            - Research doc [3] Section 7.2: "Sloppy Model Geometry"
        """
        # Define log probability based on loss type
        if loss_type == 'mse':
            def log_prob(p, sample):
                x, y = sample[0], sample[1]
                pred = model_fn(p, x)
                # Gaussian output: log p ∝ -||y - pred||²
                return -0.5 * jnp.sum((pred - y) ** 2)
        else:
            def log_prob(p, sample):
                x, y = sample[0], sample[1]
                logits = model_fn(p, x)
                # Cross-entropy: log p = y · log(softmax(logits))
                log_probs = jax.nn.log_softmax(logits)
                return jnp.sum(y * log_probs)
        
        # Package samples as (x, y) tuples
        samples = [(inputs[i], targets[i]) for i in range(len(inputs))]
        
        # Create manifold with computed Fisher
        return StatisticalManifold.from_log_prob(
            log_prob_fn=log_prob,
            params=params,
            samples=jnp.array(samples),
            detect_asymmetry=self.detect_asymmetry,
            regularization=self.regularization
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACTION METHOD 3: Pairwise Distances → Local Metric
    # ─────────────────────────────────────────────────────────────────────────
    
    def from_pairwise_distances(self,
                                 distances: jnp.ndarray,
                                 embed_dim: int = 10,
                                 symmetric: Optional[bool] = None
                                 ) -> Tuple[jnp.ndarray, Union[MetricTensor, RandersMetric]]:
        """
        Infer metric structure from pairwise distance matrix.
        
        This is for data where we only have distances/similarities,
        not raw features (e.g., graph shortest paths, kernel matrices).
        
        Algorithm:
        1. Embed points via MDS (classical or Finsler)
        2. Compute local metric from embedding
        
        Args:
            distances: (n, n) distance matrix
            embed_dim: Embedding dimension
            symmetric: If None, auto-detect; if False, use Finsler MDS
            
        Returns:
            Tuple of (embedded_points, metric)
            
        See also:
            - finsler.py: RandersMetric for asymmetric distances
            - Research doc [3] Section 4: Finsler MDS
        """
        distances = jnp.asarray(distances)
        n = distances.shape[0]
        
        # Check symmetry
        if symmetric is None:
            asymmetry = jnp.mean(jnp.abs(distances - distances.T))
            symmetric = asymmetry < 1e-6 * jnp.mean(distances)
        
        if symmetric:
            # Classical MDS
            embedding = self._classical_mds(distances, embed_dim)
            # Compute Euclidean metric at centroid
            metric = MetricTensor.euclidean(embed_dim)
        else:
            # Finsler-aware embedding (simplified)
            # First embed symmetrized distances
            sym_dist = 0.5 * (distances + distances.T)
            embedding = self._classical_mds(sym_dist, embed_dim)
            
            # Extract drift from asymmetry
            drift = self._extract_drift_from_asymmetry(distances, embedding)
            
            metric = RandersMetric(
                A=jnp.eye(embed_dim),
                b=drift
            )
        
        return embedding, metric
    
    def _classical_mds(self, distances: jnp.ndarray, k: int) -> jnp.ndarray:
        """Classical Multidimensional Scaling."""
        n = distances.shape[0]
        
        # Double centering
        D_sq = distances ** 2
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        B = -0.5 * H @ D_sq @ H
        
        # Eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(B)
        
        # Take top k positive eigenvalues
        idx = jnp.argsort(eigenvalues)[::-1][:k]
        eigenvalues = jnp.maximum(eigenvalues[idx], 0)
        eigenvectors = eigenvectors[:, idx]
        
        # Embedding
        return eigenvectors * jnp.sqrt(eigenvalues)
    
    def _extract_drift_from_asymmetry(self,
                                       distances: jnp.ndarray,
                                       embedding: jnp.ndarray) -> jnp.ndarray:
        """Extract drift vector from asymmetric distances."""
        n, k = embedding.shape
        
        # Asymmetry: d(i,j) - d(j,i)
        asymmetry = distances - distances.T
        
        # For each pair, the drift should explain the asymmetry
        # Simplified: use mean asymmetry direction
        drift_accum = jnp.zeros(k)
        
        for i in range(n):
            for j in range(i + 1, n):
                direction = embedding[j] - embedding[i]
                direction = direction / (jnp.linalg.norm(direction) + 1e-8)
                drift_accum += asymmetry[i, j] * direction
        
        # Normalize to valid Randers strength
        drift_norm = jnp.linalg.norm(drift_accum)
        if drift_norm > 0:
            return drift_accum * (0.3 / drift_norm)
        return jnp.zeros(k)
    
    # ─────────────────────────────────────────────────────────────────────────
    # VARIANCE TYPE DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def detect_variance_type(data: jnp.ndarray,
                             perturbation: jnp.ndarray) -> TensorVariance:
        """
        Determine if data transforms as vector (contravariant) or covector.
        
        The key test: apply a perturbation and see how data transforms.
        
          - Contravariant (vector): data transforms with J^{-1}
          - Covariant (covector): data transforms with J
        
        This is useful for understanding what kind of quantity your
        data represents (velocities? gradients? something else?)
        
        Args:
            data: Data samples
            perturbation: Jacobian-like transformation
            
        Returns:
            TensorVariance classification
            
        See also:
            - core.py: TensorVariance enum
            - Research doc [2] Section 2.1: "Vectors vs Covectors"
        """
        # Apply forward transformation
        forward = data @ perturbation
        
        # Apply inverse transformation
        inverse = data @ jnp.linalg.inv(perturbation)
        
        # Check which preserves structure better
        # (This is a heuristic based on norm preservation)
        original_norm = jnp.linalg.norm(data)
        forward_norm = jnp.linalg.norm(forward)
        inverse_norm = jnp.linalg.norm(inverse)
        
        # Contravariant: inverse transform preserves norm
        # Covariant: forward transform preserves norm
        if abs(inverse_norm - original_norm) < abs(forward_norm - original_norm):
            return TensorVariance.CONTRAVARIANT
        else:
            return TensorVariance.COVARIANT


# =============================================================================
# SPECIALIZED EXTRACTORS
# =============================================================================

class SPDGeometryExtractor(DataGeometryExtractor):
    """
    Specialized extractor for Symmetric Positive Definite (SPD) matrix data.
    
    SPD matrices naturally arise in:
      - Covariance estimation (EEG, MEG, fMRI)
      - Diffusion tensor imaging (DTI)
      - Radar signal processing
      - Graph Laplacians
    
    The SPD manifold has a well-known geometry:
      - Affine-invariant metric: d(A,B) = ||log(A^{-1/2} B A^{-1/2})||_F
      - Log-Euclidean metric: d(A,B) = ||log(A) - log(B)||_F
    
    The Fisher metric for Gaussian families matches the affine-invariant
    metric, validating our general approach.
    
    See also:
        - spd.py: SPDManifold class with explicit geometry
        - tests/realworld/benchmarks/learnable.py: SPDTangentSpaceModel
    """
    
    def from_spd_matrices(self,
                          matrices: jnp.ndarray,
                          metric_type: Literal['affine', 'log_euclidean'] = 'log_euclidean'
                          ) -> StatisticalManifold:
        """
        Create manifold from SPD covariance matrices.
        
        Args:
            matrices: SPD matrices (n_samples, d, d)
            metric_type: 'affine' or 'log_euclidean'
            
        Returns:
            StatisticalManifold with SPD geometry
        """
        matrices = jnp.asarray(matrices)
        n_samples, d, _ = matrices.shape
        
        # Compute Fréchet mean (simplified: use log-Euclidean mean)
        log_matrices = jax.vmap(self._safe_matrix_log)(matrices)
        mean_log = jnp.mean(log_matrices, axis=0)
        mean_matrix = self._safe_matrix_exp(mean_log)
        
        # Flatten upper triangle for parameter representation
        mean_params = self._upper_triangle(mean_matrix)
        
        # Define log probability for Wishart-like model
        # (SPD matrices as samples from centered Gaussian → Wishart)
        def log_prob(params, sample_flat):
            # Reconstruct matrix from upper triangle
            cov = self._from_upper_triangle(params, d)
            sample = sample_flat.reshape(d, d)
            
            # Log-likelihood of sample under Wishart
            cov_inv = jnp.linalg.inv(cov + self.regularization * jnp.eye(d))
            log_det = jnp.linalg.slogdet(cov)[1]
            trace_term = jnp.trace(cov_inv @ sample)
            
            return -0.5 * (trace_term + log_det)
        
        # Flatten samples
        samples_flat = matrices.reshape(n_samples, -1)
        
        return StatisticalManifold.from_log_prob(
            log_prob_fn=log_prob,
            params=mean_params,
            samples=samples_flat,
            detect_asymmetry=False,  # SPD is symmetric by definition
            regularization=self.regularization
        )
    
    def _safe_matrix_log(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Matrix logarithm with regularization."""
        eigvals, eigvecs = jnp.linalg.eigh(matrix)
        eigvals = jnp.maximum(eigvals, 1e-6)
        log_eigvals = jnp.log(eigvals)
        return eigvecs @ jnp.diag(log_eigvals) @ eigvecs.T
    
    def _safe_matrix_exp(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Matrix exponential."""
        eigvals, eigvecs = jnp.linalg.eigh(matrix)
        exp_eigvals = jnp.exp(eigvals)
        return eigvecs @ jnp.diag(exp_eigvals) @ eigvecs.T
    
    def _upper_triangle(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Extract upper triangle as flat vector."""
        d = matrix.shape[0]
        indices = jnp.triu_indices(d)
        return matrix[indices]
    
    def _from_upper_triangle(self, flat: jnp.ndarray, d: int) -> jnp.ndarray:
        """Reconstruct symmetric matrix from upper triangle."""
        matrix = jnp.zeros((d, d))
        indices = jnp.triu_indices(d)
        matrix = matrix.at[indices].set(flat)
        matrix = matrix + matrix.T - jnp.diag(jnp.diag(matrix))
        return matrix


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataGeometryExtractor',
    'SPDGeometryExtractor',
]

