"""
Statistical Manifold: Universal Geometric Structure from Data

This module implements the core abstraction for treating any parametric model
as a point on a statistical manifold, with Fisher Information as the natural
Riemannian metric.

=============================================================================
THEORY (from research docs [2] and [3]):
=============================================================================

The key insight is that ANY dataset defines a statistical manifold:

    g_ij(θ) = E[∂_i log p(x;θ) · ∂_j log p(x;θ)]

This Fisher Information Matrix is:
  - The UNIQUE Riemannian metric invariant to sufficient statistics
  - The natural measure of "distinguishability" between distributions
  - The Hessian of KL divergence (locally)

This is UNIVERSAL - it doesn't care if your data is:
  - Time series → temporal covariance emerges
  - Images → spatial correlations emerge
  - Graphs → structural patterns emerge
  - Mixed data → joint distributions

=============================================================================
CONNECTIONS TO OTHER MODULES:
=============================================================================

  ┌─────────────────────────────────────────────────────────────────────┐
  │  statistical_manifold.py  (THIS FILE)                               │
  │    ↓ provides Fisher metric                                         │
  │  information.py  →  FisherMetric class (extends MetricTensor)       │
  │    ↓ used by                                                        │
  │  geometry_extractor.py  →  Extracts geometry from raw data          │
  │    ↓ enables                                                        │
  │  optimizer.py  →  Duality-aware gradient descent                    │
  └─────────────────────────────────────────────────────────────────────┘

See also:
  - core.py: TensorVariance, Parity, GeometricSignature
  - metric.py: MetricTensor base class
  - finsler.py: RandersMetric for asymmetric extension

Reference: Amari "Information Geometry and Its Applications" (2016)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union, Any
from functools import partial

# ─────────────────────────────────────────────────────────────────────────────
# Internal imports - see these files for related concepts
# ─────────────────────────────────────────────────────────────────────────────
from ..core import TensorVariance, MetricType, GeometricSignature
from ..geometry.metric import MetricTensor
from .fisher import FisherMetric
from ..geometry.finsler import RandersMetric


# =============================================================================
# STATISTICAL MANIFOLD: Core Abstraction
# =============================================================================

@dataclass
class StatisticalManifold:
    """
    A statistical manifold defined by a parametric family p(x|θ).
    
    This is the UNIVERSAL abstraction for geometric pattern mining:
    any model that assigns probabilities to observations defines a manifold.
    
    The geometry is determined by the Fisher Information, which measures
    how "informative" observations are about parameters. High curvature
    means parameters are well-constrained; low curvature means "sloppy"
    directions where data provides little information.
    
    ==========================================================================
    USAGE PATTERN:
    ==========================================================================
    
        # 1. Define log probability function
        def log_prob(params, x):
            mu, sigma = params
            return -0.5 * ((x - mu) / sigma)**2 - jnp.log(sigma)
        
        # 2. Create manifold at current parameters
        manifold = StatisticalManifold.from_log_prob(log_prob, params, samples)
        
        # 3. Use Fisher metric for optimization
        natural_grad = manifold.natural_gradient(euclidean_grad)
    
    ==========================================================================
    ATTRIBUTES:
    ==========================================================================
    
    log_prob_fn : Callable[[params, x], float]
        Log probability function defining the distribution family.
        This is the CORE definition - everything else derives from this.
        
    params : jnp.ndarray
        Current parameter values θ. The manifold is a space of ALL possible
        θ values; this specifies "where we are" on that space.
        
    fisher_metric : FisherMetric
        The Riemannian metric at current params. This is what makes
        gradient descent "geometric" - it accounts for curvature.
        
    asymmetric_component : Optional[jnp.ndarray]
        If data shows directional bias (non-zero skewness), this drift
        vector enables Finsler extension. See finsler.py for details.
    
    ==========================================================================
    CONNECTION TO EXISTING CODE:
    ==========================================================================
    
    This class builds on:
      - FisherMetric (information.py): Computes g_ij from samples
      - MetricTensor (metric.py): Base class for Riemannian metrics
      - RandersMetric (finsler.py): Extension for asymmetric data
    """
    
    # Core definition: log probability function
    log_prob_fn: Callable[[Any, jnp.ndarray], float]
    
    # Current position on manifold
    params: jnp.ndarray
    
    # Computed geometry (lazily evaluated)
    _fisher_metric: Optional[FisherMetric] = field(default=None, repr=False)
    _samples: Optional[jnp.ndarray] = field(default=None, repr=False)
    
    # Finsler extension for asymmetric data
    asymmetric_component: Optional[jnp.ndarray] = None
    
    # Configuration
    regularization: float = 1e-6
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCTION METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def from_log_prob(cls,
                      log_prob_fn: Callable[[Any, jnp.ndarray], float],
                      params: jnp.ndarray,
                      samples: jnp.ndarray,
                      detect_asymmetry: bool = True,
                      regularization: float = 1e-6) -> 'StatisticalManifold':
        """
        Create manifold from log probability function and data samples.
        
        This is the PRIMARY constructor - it computes the Fisher metric
        from the score function (gradient of log prob w.r.t. params).
        
        Args:
            log_prob_fn: Function (params, x) -> log p(x|params)
            params: Current parameter values
            samples: Data samples for Monte Carlo estimation of Fisher
            detect_asymmetry: If True, check for directional bias
            regularization: Diagonal regularization for numerical stability
            
        Returns:
            StatisticalManifold with computed Fisher metric
            
        Example:
            >>> def gaussian_log_prob(theta, x):
            ...     return -0.5 * jnp.sum((x - theta)**2)
            >>> manifold = StatisticalManifold.from_log_prob(
            ...     gaussian_log_prob, mu, samples
            ... )
        """
        manifold = cls(
            log_prob_fn=log_prob_fn,
            params=params,
            _samples=samples,
            regularization=regularization
        )
        
        # Compute Fisher metric from samples
        manifold._compute_fisher_metric()
        
        # Check for asymmetry (Finsler extension)
        if detect_asymmetry:
            manifold._detect_asymmetry()
        
        return manifold
    
    @classmethod
    def from_gaussian(cls,
                      mean: jnp.ndarray,
                      covariance: jnp.ndarray,
                      samples: Optional[jnp.ndarray] = None) -> 'StatisticalManifold':
        """
        Create manifold for Gaussian family N(μ, Σ).
        
        For Gaussians, the Fisher metric has a known analytical form:
          - For mean parameters: F_μ = Σ^{-1}
          - For full (μ, Σ): Block structure with Wishart term
        
        This is useful for testing and for SPD data (covariance matrices).
        
        Args:
            mean: Mean vector μ
            covariance: Covariance matrix Σ
            samples: Optional samples (not needed for analytical Fisher)
            
        Returns:
            StatisticalManifold with analytical Fisher metric
        """
        dim = mean.shape[0]
        
        # Analytical Fisher for Gaussian mean
        cov_inv = jnp.linalg.inv(covariance + 1e-6 * jnp.eye(dim))
        fisher = FisherMetric(cov_inv, params=mean)
        
        # Define the log prob function for completeness
        def log_prob(params, x):
            diff = x - params
            return -0.5 * diff @ cov_inv @ diff
        
        manifold = cls(
            log_prob_fn=log_prob,
            params=mean,
            _samples=samples
        )
        manifold._fisher_metric = fisher
        
        return manifold
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE GEOMETRIC OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def fisher_metric(self) -> FisherMetric:
        """
        The Fisher Information Matrix as a Riemannian metric.
        
        This is THE key object: it defines the geometry of parameter space.
        Used for:
          - Natural gradient: F^{-1} ∇L
          - Distance between models: √(δθ^T F δθ)
          - Cramér-Rao bound: Var(θ̂) ≥ F^{-1}
        
        Returns:
            FisherMetric instance (see information.py)
        """
        if self._fisher_metric is None:
            self._compute_fisher_metric()
        return self._fisher_metric
    
    @property
    def metric_type(self) -> MetricType:
        """Classification of the metric structure."""
        if self.asymmetric_component is not None:
            return MetricType.FINSLER
        return MetricType.RIEMANNIAN
    
    @property
    def dim(self) -> int:
        """Dimension of parameter space."""
        return self.params.shape[0] if self.params.ndim > 0 else 1
    
    def natural_gradient(self, euclidean_gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Convert Euclidean gradient to natural gradient.
        
        THE KEY OPERATION for geometric optimization:
        
            ∇_nat L = F^{-1} ∇L
        
        Geometrically: the Euclidean gradient ∇L is a COVECTOR (lives in
        dual/cotangent space). The natural gradient is the corresponding
        VECTOR (lives in tangent space). The Fisher metric provides the
        isomorphism between these spaces.
        
        Why this matters:
          - Euclidean gradient: direction of steepest descent in L2
          - Natural gradient: steepest descent in KL divergence
          - Natural gradient is invariant to reparameterization
        
        Args:
            euclidean_gradient: ∇L (covector, covariant)
            
        Returns:
            F^{-1} ∇L (vector, contravariant)
            
        See also:
            - core.py: TensorVariance.COVARIANT vs CONTRAVARIANT
            - metric.py: MetricTensor.raise_index()
        """
        return self.fisher_metric.natural_gradient(euclidean_gradient)
    
    def geodesic_distance(self, other_params: jnp.ndarray) -> float:
        """
        Approximate geodesic distance to another point on manifold.
        
        For small displacements δθ:
            d(θ, θ + δθ) ≈ √(δθ^T F δθ)
        
        This is the Riemannian distance induced by Fisher metric.
        """
        delta = other_params - self.params
        return float(self.fisher_metric.norm(delta))
    
    def kl_divergence_local(self, delta_params: jnp.ndarray) -> float:
        """
        Local KL divergence approximation.
        
        For small δθ:
            KL(p_θ || p_{θ+δ}) ≈ (1/2) δθ^T F δθ
        
        The Fisher metric is the Hessian of KL divergence at θ.
        """
        return self.fisher_metric.kl_divergence_local(delta_params)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINSLER EXTENSION (for asymmetric data)
    # ─────────────────────────────────────────────────────────────────────────
    
    def as_randers_metric(self) -> Optional[RandersMetric]:
        """
        Get Randers (Finsler) metric if asymmetry detected.
        
        A Randers metric is: F(v) = √(v^T A v) + b^T v
        
        The drift vector b captures directional bias in the data:
          - Moving "with the flow" is cheaper
          - Moving "against" is more expensive
        
        Returns:
            RandersMetric if asymmetric_component is set, else None
            
        See also:
            - finsler.py: RandersMetric class
            - Design doc Section 2.2 on Finsler geometry
        """
        if self.asymmetric_component is None:
            return None
        
        return RandersMetric(
            A=self.fisher_metric.matrix,
            b=self.asymmetric_component
        )
    
    def is_asymmetric(self) -> bool:
        """Check if manifold has directional bias (Finsler structure)."""
        return self.asymmetric_component is not None
    
    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL COMPUTATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_fisher_metric(self) -> None:
        """
        Compute Fisher Information from score function.
        
        The score function is s(x) = ∇_θ log p(x|θ).
        Fisher Information is the covariance of scores:
        
            F_ij = E[s_i · s_j] = Cov(s)
        
        We estimate this via Monte Carlo over samples.
        """
        if self._samples is None:
            raise ValueError(
                "Cannot compute Fisher metric without samples. "
                "Either provide samples or use from_gaussian() for analytical form."
            )
        
        # Use FisherMetric.from_log_likelihood for the heavy lifting
        # (see information.py for implementation)
        self._fisher_metric = FisherMetric.from_log_likelihood(
            log_likelihood_fn=self.log_prob_fn,
            params=self.params,
            samples=self._samples,
            regularization=self.regularization
        )
    
    def _detect_asymmetry(self) -> None:
        """
        Detect directional bias in score function (third moment).
        
        If E[s_i s_j s_k] ≠ 0, there's asymmetry that the symmetric
        Fisher metric can't capture. We extract the dominant asymmetric
        direction as a drift vector for Finsler extension.
        
        Mathematically: we compute the skewness of score vectors.
        If significant, the first principal component of the third
        moment tensor becomes the drift vector b.
        """
        if self._samples is None:
            return
        
        # Compute score function for each sample
        score_fn = jax.grad(self.log_prob_fn, argnums=0)
        
        try:
            scores = jax.vmap(lambda x: score_fn(self.params, x))(self._samples)
        except Exception:
            # If score computation fails, skip asymmetry detection
            return
        
        # Compute mean score (should be ~0 for well-specified models)
        mean_score = jnp.mean(scores, axis=0)
        centered_scores = scores - mean_score
        
        # Compute third moment (skewness direction)
        # We look for E[s · ||s||²] which gives dominant asymmetric direction
        score_norms_sq = jnp.sum(centered_scores ** 2, axis=1, keepdims=True)
        weighted_scores = centered_scores * score_norms_sq
        third_moment_direction = jnp.mean(weighted_scores, axis=0)
        
        # Check if asymmetry is significant
        asymmetry_strength = jnp.linalg.norm(third_moment_direction)
        fisher_scale = jnp.sqrt(jnp.trace(self.fisher_metric.matrix) / self.dim)
        
        # Threshold: asymmetry must be > 10% of Fisher scale
        if asymmetry_strength > 0.1 * fisher_scale:
            # Normalize to valid Randers drift (must satisfy |b|_A < 1)
            A_inv = self.fisher_metric.inverse
            b_norm_A = jnp.sqrt(third_moment_direction @ A_inv @ third_moment_direction)
            
            if b_norm_A > 0:
                # Scale to 0.3 strength (conservative Finsler)
                self.asymmetric_component = third_moment_direction * (0.3 / (b_norm_A + 1e-8))
    
    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_params(self, new_params: jnp.ndarray, 
                      new_samples: Optional[jnp.ndarray] = None) -> 'StatisticalManifold':
        """
        Create new manifold at updated parameter values.
        
        The Fisher metric depends on θ, so moving on the manifold
        requires recomputing the metric at the new location.
        
        Args:
            new_params: New parameter values
            new_samples: New samples (optional, reuses old if not provided)
            
        Returns:
            New StatisticalManifold at updated position
        """
        samples = new_samples if new_samples is not None else self._samples
        
        return StatisticalManifold.from_log_prob(
            log_prob_fn=self.log_prob_fn,
            params=new_params,
            samples=samples,
            detect_asymmetry=self.asymmetric_component is not None,
            regularization=self.regularization
        )
    
    def __repr__(self) -> str:
        metric_str = "Finsler" if self.is_asymmetric() else "Riemannian"
        return f"StatisticalManifold(dim={self.dim}, metric={metric_str})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def empirical_fisher_from_data(
    model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray
) -> FisherMetric:
    """
    Compute empirical Fisher from a predictive model.
    
    For neural networks and similar models where we have:
      - model_fn(params, x) → predictions
      - Cross-entropy or MSE loss against targets
    
    The empirical Fisher uses the model's own predictions:
        F_emp = E_{x~data}[∇log p(y|x,θ) ∇log p(y|x,θ)^T]
    
    This is what optimizers like Adam implicitly approximate.
    
    Args:
        model_fn: Function (params, x) -> predictions
        params: Model parameters (flattened)
        inputs: Input data batch
        targets: Target values
        
    Returns:
        FisherMetric estimated from gradients
        
    See also:
        - information.py: FisherAtom.empirical_fisher() for layer-wise
    """
    def log_prob(p, x_and_y):
        x, y = x_and_y[:x_and_y.shape[0]//2], x_and_y[x_and_y.shape[0]//2:]
        pred = model_fn(p, x)
        # Assume Gaussian output for simplicity
        return -0.5 * jnp.sum((pred - y) ** 2)
    
    # Stack inputs and targets for vmap
    samples = jnp.concatenate([inputs, targets], axis=-1)
    
    return FisherMetric.from_log_likelihood(
        log_likelihood_fn=log_prob,
        params=params,
        samples=samples
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StatisticalManifold',
    'empirical_fisher_from_data',
]

