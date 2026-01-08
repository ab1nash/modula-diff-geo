"""
Geometric Optimizer: Duality-Aware Gradient Descent

This module implements optimization that properly respects the geometric
distinction between gradients (covectors) and updates (vectors).

=============================================================================
THE FUNDAMENTAL INSIGHT (from research doc [2]):
=============================================================================

In standard gradient descent, we write:
    θ_new = θ - lr * ∇L

This is GEOMETRICALLY WRONG! Here's why:

  - θ (parameters) are VECTORS: they represent positions
  - ∇L (gradients) are COVECTORS: they live in the dual space
  
You cannot subtract a covector from a vector - they're different types!
The operation only makes sense if there's an isomorphism between them.

In Euclidean space (identity metric), vectors = covectors, so we don't
notice the problem. But on curved manifolds, we MUST convert:

    δθ = g^{-1}(∇L)     # Raise index: covector → vector
    θ_new = θ - lr * δθ  # Now both are vectors!

This is exactly what natural gradient does with Fisher metric.

=============================================================================
CONNECTIONS:
=============================================================================

  ┌─────────────────────────────────────────────────────────────────────┐
  │  GRADIENT ∇L                                                        │
  │    ↓ (lives in T*M, cotangent/dual space)                          │
  │                                                                     │
  │  optimizer.py (THIS FILE)                                           │
  │    ↓ uses metric to convert                                        │
  │                                                                     │
  │  statistical_manifold.py → StatisticalManifold.natural_gradient()   │
  │    ↓ or                                                            │
  │  finsler.py → FinslerDualizer.dualize()                            │
  │    ↓                                                               │
  │                                                                     │
  │  UPDATE δθ                                                          │
  │    (lives in TM, tangent space - same as parameters!)              │
  └─────────────────────────────────────────────────────────────────────┘

See also:
  - core.py: TensorVariance.COVARIANT (gradients) vs CONTRAVARIANT (updates)
  - metric.py: MetricTensor.raise_index() - the mathematical operation
  - information.py: FisherMetric for natural gradient
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, Any, Union
from functools import partial

# ─────────────────────────────────────────────────────────────────────────────
# Internal imports
# ─────────────────────────────────────────────────────────────────────────────
from ..core import TensorVariance, MetricType
from ..geometry.metric import MetricTensor
from ..information.fisher import FisherMetric
from ..geometry.finsler import RandersMetric, FinslerDualizer
from ..information.manifolds import StatisticalManifold


# =============================================================================
# GEOMETRIC OPTIMIZER STATE
# =============================================================================

@dataclass
class GeometricOptimizerState:
    """
    State for geometric optimizer (analogous to Adam state).
    
    Tracks:
      - Current position on manifold (params)
      - Momentum in tangent space (if using momentum)
      - Fisher metric history (for online estimation)
      - Step count
    
    Note: Momentum must be PARALLEL TRANSPORTED between steps on curved
    manifolds. For now, we use a first-order approximation (just scale).
    """
    params: jnp.ndarray
    step: int = 0
    momentum: Optional[jnp.ndarray] = None
    fisher_ema: Optional[jnp.ndarray] = None  # Exponential moving average of Fisher
    
    # Configuration
    momentum_decay: float = 0.9
    fisher_ema_decay: float = 0.99


# =============================================================================
# GEOMETRIC OPTIMIZER: Main Class
# =============================================================================

class GeometricOptimizer:
    """
    Optimizer that respects covariant/contravariant distinction.
    
    This is the geometric equivalent of Adam/SGD: it performs gradient
    descent, but properly converts gradients (covectors) to updates
    (vectors) using the manifold's metric.
    
    ==========================================================================
    THE KEY OPERATION:
    ==========================================================================
    
    Standard SGD:    θ_new = θ - lr * ∇L
    Geometric:       θ_new = θ - lr * g^{-1}(∇L)
    
    The g^{-1} is "index raising" - converting a covector to a vector
    using the metric tensor. For Fisher metric, this gives natural gradient.
    
    ==========================================================================
    USAGE:
    ==========================================================================
    
        # Create optimizer with a statistical manifold
        manifold = StatisticalManifold.from_log_prob(log_prob, params, data)
        optimizer = GeometricOptimizer(manifold, lr=0.01)
        
        # Training loop
        state = optimizer.init(params)
        for batch in dataloader:
            grad = compute_gradient(state.params, batch)
            state = optimizer.step(state, grad)
    
    ==========================================================================
    ATTRIBUTES:
    ==========================================================================
    
    manifold : StatisticalManifold
        Defines the geometry (Fisher metric). If None, uses Euclidean.
        
    learning_rate : float
        Step size in tangent space.
        
    use_momentum : bool
        Whether to use momentum (with parallel transport).
        
    adaptive_metric : bool
        Whether to update Fisher metric online during training.
    """
    
    def __init__(self,
                 manifold: Optional[StatisticalManifold] = None,
                 learning_rate: float = 0.01,
                 use_momentum: bool = False,
                 momentum_decay: float = 0.9,
                 adaptive_metric: bool = False,
                 fisher_ema_decay: float = 0.99):
        """
        Initialize geometric optimizer.
        
        Args:
            manifold: StatisticalManifold defining the geometry
            learning_rate: Step size
            use_momentum: Whether to use momentum
            momentum_decay: Decay rate for momentum
            adaptive_metric: Whether to update metric during training
            fisher_ema_decay: Decay for online Fisher estimation
        """
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.use_momentum = use_momentum
        self.momentum_decay = momentum_decay
        self.adaptive_metric = adaptive_metric
        self.fisher_ema_decay = fisher_ema_decay
    
    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def init(self, params: jnp.ndarray) -> GeometricOptimizerState:
        """
        Initialize optimizer state.
        
        Args:
            params: Initial parameter values
            
        Returns:
            GeometricOptimizerState
        """
        state = GeometricOptimizerState(
            params=params,
            step=0,
            momentum_decay=self.momentum_decay,
            fisher_ema_decay=self.fisher_ema_decay
        )
        
        if self.use_momentum:
            state.momentum = jnp.zeros_like(params)
        
        if self.adaptive_metric and self.manifold is not None:
            state.fisher_ema = self.manifold.fisher_metric.matrix.copy()
        
        return state
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE STEP: The geometric update
    # ─────────────────────────────────────────────────────────────────────────
    
    def step(self,
             state: GeometricOptimizerState,
             gradient: jnp.ndarray,
             samples: Optional[jnp.ndarray] = None
             ) -> GeometricOptimizerState:
        """
        Perform one optimization step.
        
        THE GEOMETRIC OPERATION:
        
        1. gradient is a COVECTOR (lives in T*M, dual space)
        2. Convert to VECTOR using metric: update = g^{-1}(gradient)
        3. Apply momentum (if enabled) in tangent space
        4. Update parameters: θ_new = θ - lr * update
        
        Args:
            state: Current optimizer state
            gradient: Euclidean gradient ∇L (a covector!)
            samples: New samples for adaptive metric update
            
        Returns:
            Updated GeometricOptimizerState
        """
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Convert gradient (covector) to update (vector)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.manifold is not None:
            # Use manifold geometry
            if self.manifold.is_asymmetric():
                # Finsler case: use RandersMetric dualizer
                randers = self.manifold.as_randers_metric()
                dualizer = FinslerDualizer(randers)
                update = dualizer.dualize(gradient)
            else:
                # Riemannian case: natural gradient
                update = self.manifold.natural_gradient(gradient)
        else:
            # Euclidean case: gradient = update (identity metric)
            update = gradient
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Apply momentum (in tangent space)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.use_momentum and state.momentum is not None:
            # Parallel transport approximation: just scale
            # (Proper parallel transport would require Christoffel symbols)
            momentum = state.momentum_decay * state.momentum + update
            update = momentum
        else:
            momentum = state.momentum
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Update parameters
        # ─────────────────────────────────────────────────────────────────────
        
        new_params = state.params - self.learning_rate * update
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Update Fisher metric (if adaptive)
        # ─────────────────────────────────────────────────────────────────────
        
        fisher_ema = state.fisher_ema
        if self.adaptive_metric and samples is not None and self.manifold is not None:
            fisher_ema = self._update_fisher_ema(
                state.fisher_ema, gradient, state.fisher_ema_decay
            )
        
        return GeometricOptimizerState(
            params=new_params,
            step=state.step + 1,
            momentum=momentum,
            fisher_ema=fisher_ema,
            momentum_decay=state.momentum_decay,
            fisher_ema_decay=state.fisher_ema_decay
        )
    
    def _update_fisher_ema(self,
                           fisher_ema: jnp.ndarray,
                           gradient: jnp.ndarray,
                           decay: float) -> jnp.ndarray:
        """
        Update Fisher estimate via exponential moving average.
        
        Approximation: F ≈ E[∇L ∇L^T]
        This is the "empirical Fisher" used in Adam-like optimizers.
        """
        outer = jnp.outer(gradient, gradient)
        return decay * fisher_ema + (1 - decay) * outer
    
    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_manifold(self,
                        new_manifold: StatisticalManifold) -> 'GeometricOptimizer':
        """
        Update the manifold (e.g., after moving to new region).
        
        The Fisher metric depends on position, so we may need to
        recompute it as we move through parameter space.
        """
        return GeometricOptimizer(
            manifold=new_manifold,
            learning_rate=self.learning_rate,
            use_momentum=self.use_momentum,
            momentum_decay=self.momentum_decay,
            adaptive_metric=self.adaptive_metric,
            fisher_ema_decay=self.fisher_ema_decay
        )


# =============================================================================
# SPECIALIZED OPTIMIZERS
# =============================================================================

class NaturalGradientOptimizer(GeometricOptimizer):
    """
    Natural Gradient Descent with Fisher Information.
    
    This is the classic "natural gradient" from Amari:
        θ_new = θ - lr * F^{-1}(θ) ∇L
    
    The Fisher inverse converts the gradient from dual space to
    tangent space, giving the steepest descent in KL divergence.
    
    Advantages over SGD:
      - Invariant to reparameterization
      - Often faster convergence
      - Better conditioning for ill-conditioned problems
    
    See also:
      - information.py: FisherMetric.natural_gradient()
      - Research doc [3] Section 3.1: "The Statistical Manifold and Fisher Information"
    """
    
    def __init__(self,
                 manifold: StatisticalManifold,
                 learning_rate: float = 0.01):
        """
        Initialize natural gradient optimizer.
        
        Args:
            manifold: Must have Fisher metric computed
            learning_rate: Step size
        """
        super().__init__(
            manifold=manifold,
            learning_rate=learning_rate,
            use_momentum=False,
            adaptive_metric=False
        )


class FinslerOptimizer(GeometricOptimizer):
    """
    Optimizer for asymmetric (Finsler) manifolds.
    
    When the data has directional bias (non-zero skewness), the
    symmetric Fisher metric doesn't capture the full geometry.
    The Finsler optimizer uses a Randers metric:
    
        F(v) = √(v^T A v) + b^T v
    
    where b is the drift vector. This makes:
      - Moving "with the flow" cheaper
      - Moving "against" more expensive
    
    Useful for:
      - Directed graphs (social networks, citations)
      - Causal modeling
      - Thermodynamic irreversibility
    
    See also:
      - finsler.py: RandersMetric, FinslerDualizer
      - Design doc Section 2.2: "Finsler Geometry and Asymmetry"
    """
    
    def __init__(self,
                 manifold: StatisticalManifold,
                 learning_rate: float = 0.01,
                 drift_strength: float = 0.3):
        """
        Initialize Finsler optimizer.
        
        Args:
            manifold: Should have asymmetric_component set
            learning_rate: Step size
            drift_strength: Scaling for the asymmetric component
        """
        # Ensure manifold has Finsler structure
        if manifold.asymmetric_component is None:
            # Add small drift for regularization
            manifold.asymmetric_component = jnp.zeros(manifold.dim) + 0.01
        
        super().__init__(
            manifold=manifold,
            learning_rate=learning_rate,
            use_momentum=False,
            adaptive_metric=False
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def geometric_sgd_step(params: jnp.ndarray,
                       gradient: jnp.ndarray,
                       metric: Union[MetricTensor, FisherMetric],
                       learning_rate: float = 0.01) -> jnp.ndarray:
    """
    Single geometric SGD step (functional interface).
    
    This is for users who want a simple function rather than a class.
    
    Args:
        params: Current parameters
        gradient: Euclidean gradient (covector)
        metric: Metric tensor for index raising
        learning_rate: Step size
        
    Returns:
        Updated parameters
    """
    # Convert gradient to update vector
    update = metric.raise_index(gradient)
    
    # Apply update
    return params - learning_rate * update


def natural_gradient_step(params: jnp.ndarray,
                          gradient: jnp.ndarray,
                          fisher: FisherMetric,
                          learning_rate: float = 0.01) -> jnp.ndarray:
    """
    Single natural gradient step (functional interface).
    
    Convenience wrapper around geometric_sgd_step with FisherMetric.
    """
    return geometric_sgd_step(params, gradient, fisher, learning_rate)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'GeometricOptimizerState',
    'GeometricOptimizer',
    'NaturalGradientOptimizer',
    'FinslerOptimizer',
    'geometric_sgd_step',
    'natural_gradient_step',
]

