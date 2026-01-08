"""
Fisher Information Geometry

Implements the Fisher Information Matrix as a Riemannian metric on
statistical manifolds, enabling natural gradient descent and geometric
analysis of probability distributions.

From the research document Section 4:
"In the fields of Statistics, Signal Processing, and Brain-Computer Interfaces,
the term 'covariance' takes on a double meaning... these covariance matrices
are treated as geometric objects residing on a curved manifold."

Key concepts:
- Fisher Information: g_ij(θ) = E[∂_i log p · ∂_j log p]
- Natural Gradient: ∇_nat L = F^{-1} ∇L
- Cramér-Rao Bound: Var(θ̂) ≥ F^{-1}

Reference: Amari, S. (2016). "Information Geometry and Its Applications"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, List
from functools import partial

from ..geometry.metric import MetricTensor


class FisherMetric(MetricTensor):
    """
    Fisher Information as a Riemannian metric on statistical manifolds.
    
    The Fisher Information Matrix (FIM) defines the natural geometry of
    parameter space for a family of probability distributions p(x|θ).
    
    Properties:
    - Positive semi-definite (definite for regular families)
    - Invariant under sufficient statistics
    - Provides lower bound on estimator variance (Cramér-Rao)
    
    Mathematical definition:
        F_ij(θ) = E_p[∂_i log p(x|θ) · ∂_j log p(x|θ)]
                = -E_p[∂_i ∂_j log p(x|θ)]  (under regularity conditions)
    """
    
    def __init__(self, 
                 fisher_matrix: jnp.ndarray,
                 params: Optional[jnp.ndarray] = None,
                 regularization: float = 1e-6):
        """
        Initialize Fisher metric from a precomputed Fisher matrix.
        
        Args:
            fisher_matrix: The Fisher Information Matrix F_ij
            params: Current parameter values θ (for reference)
            regularization: Small value added to diagonal for numerical stability
        """
        # Add regularization for numerical stability
        regularized = fisher_matrix + regularization * jnp.eye(fisher_matrix.shape[0])
        super().__init__(regularized)
        
        self._params = params
        self._regularization = regularization
    
    @classmethod
    def from_log_likelihood(cls,
                            log_likelihood_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
                            params: jnp.ndarray,
                            samples: jnp.ndarray,
                            regularization: float = 1e-6) -> 'FisherMetric':
        """
        Compute Fisher metric from log-likelihood function via sampling.
        
        The Fisher Information is estimated as:
            F_ij ≈ (1/N) Σ_n ∂_i log p(x_n|θ) · ∂_j log p(x_n|θ)
        
        Args:
            log_likelihood_fn: Function (params, x) -> log p(x|params)
            params: Current parameter values θ
            samples: Data samples x_1, ..., x_N
            regularization: Diagonal regularization
            
        Returns:
            FisherMetric instance
        """
        # Compute score function (gradient of log-likelihood w.r.t. params)
        score_fn = jax.grad(log_likelihood_fn, argnums=0)
        
        # Compute scores for all samples
        scores = jax.vmap(lambda x: score_fn(params, x))(samples)
        
        # Fisher matrix is outer product of scores averaged over samples
        # F_ij = E[s_i · s_j] where s = ∇_θ log p
        fisher_matrix = jnp.mean(
            jax.vmap(lambda s: jnp.outer(s, s))(scores),
            axis=0
        )
        
        return cls(fisher_matrix, params, regularization)
    
    @classmethod
    def from_hessian(cls,
                     log_likelihood_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
                     params: jnp.ndarray,
                     samples: jnp.ndarray,
                     regularization: float = 1e-6) -> 'FisherMetric':
        """
        Compute Fisher metric from negative expected Hessian.
        
        Under regularity conditions:
            F_ij = -E[∂_i ∂_j log p(x|θ)]
        
        This can be more stable than the score-based estimate.
        """
        # Compute Hessian of log-likelihood
        hessian_fn = jax.hessian(log_likelihood_fn, argnums=0)
        
        # Average negative Hessian over samples
        hessians = jax.vmap(lambda x: hessian_fn(params, x))(samples)
        fisher_matrix = -jnp.mean(hessians, axis=0)
        
        return cls(fisher_matrix, params, regularization)
    
    @property
    def params(self) -> Optional[jnp.ndarray]:
        """Current parameter values."""
        return self._params
    
    def natural_gradient(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Compute natural gradient: F^{-1} ∇L
        
        The natural gradient is the steepest descent direction in the
        Fisher-Rao geometry. It's invariant to reparameterization and
        often leads to faster convergence than vanilla gradients.
        
        Args:
            gradient: Euclidean gradient ∇L
            
        Returns:
            Natural gradient F^{-1} ∇L
        """
        return self.raise_index(gradient)
    
    @property
    def cramer_rao_bound(self) -> jnp.ndarray:
        """
        Compute Cramér-Rao lower bound on estimator variance.
        
        For any unbiased estimator θ̂ of θ:
            Var(θ̂) ≥ F^{-1}(θ)
        
        Returns:
            F^{-1}, the lower bound on variance matrix
        """
        return self.inverse
    
    def kl_divergence_local(self, delta_params: jnp.ndarray) -> float:
        """
        Local KL divergence approximation using Fisher metric.
        
        For small parameter changes δθ:
            KL(p_θ || p_{θ+δ}) ≈ (1/2) δθ^T F δθ
        
        This is the Riemannian distance squared in the Fisher geometry.
        """
        return 0.5 * self.norm(delta_params) ** 2
    
    def geodesic_update(self, 
                        gradient: jnp.ndarray, 
                        step_size: float) -> jnp.ndarray:
        """
        Compute geodesic update in parameter space.
        
        In flat coordinates (which the Fisher metric is not, in general),
        this reduces to: θ_new = θ - step_size * F^{-1} ∇L
        
        Args:
            gradient: Loss gradient ∇L
            step_size: Learning rate
            
        Returns:
            Parameter update δθ
        """
        nat_grad = self.natural_gradient(gradient)
        return -step_size * nat_grad
    
    # =========================================================================
    # DIAGONAL APPROXIMATION - O(n) operations
    # =========================================================================
    
    def diagonal(self) -> jnp.ndarray:
        """
        Return diagonal of Fisher matrix.
        
        This is O(n) storage and operations, compared to O(n²) for full matrix.
        The diagonal approximation is what Adam implicitly uses via its
        second moment estimation.
        
        Returns:
            1D array of diagonal entries
            
        Complexity: O(n)
        """
        return jnp.diag(self.matrix)
    
    def natural_gradient_diagonal(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """
        Diagonal natural gradient approximation.
        
        Instead of F^{-1} g (which is O(n³) for matrix inverse),
        computes g_i / F_ii (which is O(n)).
        
        This is the core operation behind Adam-style optimizers:
            update_i = g_i / sqrt(v_i)
        where v_i is the running average of g_i².
        
        Args:
            gradient: Euclidean gradient ∇L
            
        Returns:
            Diagonal-approximated natural gradient
            
        Complexity: O(n)
        """
        diag = jnp.maximum(jnp.diag(self.matrix), 1e-8)
        return gradient / diag
    
    # =========================================================================
    # SLOPPY MODEL ANALYSIS - Lightweight diagnostics
    # =========================================================================
    
    def effective_dimension(self, threshold: float = 1e-3) -> int:
        """
        Count eigenvalues above threshold - identifies 'stiff' directions.
        
        In "sloppy" models (common in biology, deep learning), most
        eigenvalues are very small. The effective dimension counts
        how many directions are actually constrained by the data.
        
        Reference: Machta et al. "Parameter Space Compression Underlies
        Emergent Theories and Predictive Models" (Science 2013)
        
        Args:
            threshold: Fraction of max eigenvalue to count as "stiff"
            
        Returns:
            Number of eigenvalues above threshold * max_eigenvalue
            
        Complexity: O(n²) for symmetric eigendecomposition (one-time)
        """
        eigvals = jnp.linalg.eigvalsh(self.matrix)
        cutoff = threshold * jnp.max(eigvals)
        return int(jnp.sum(eigvals > cutoff))
    
    def condition_number(self) -> float:
        """
        Ratio of max/min eigenvalue - measures 'sloppiness'.
        
        High condition number (e.g., > 10⁶) indicates:
        - The model is "sloppy" with many flat directions
        - Gradient descent will be slow
        - Natural gradient becomes important
        
        Returns:
            Condition number κ = λ_max / λ_min
            
        Complexity: O(n²) for symmetric eigendecomposition
        """
        eigvals = jnp.linalg.eigvalsh(self.matrix)
        min_eigval = jnp.min(jnp.abs(eigvals))
        max_eigval = jnp.max(jnp.abs(eigvals))
        return float(max_eigval / (min_eigval + 1e-10))
    
    def eigenspectrum(self) -> jnp.ndarray:
        """
        Return sorted eigenvalues of Fisher matrix.
        
        Useful for visualizing the "sloppiness" of a model.
        Eigenvalues typically span many orders of magnitude.
        
        Returns:
            Array of eigenvalues, sorted descending
            
        Complexity: O(n²) for symmetric eigendecomposition
        """
        eigvals = jnp.linalg.eigvalsh(self.matrix)
        return jnp.sort(eigvals)[::-1]  # Descending order
    
    def stiff_directions(self, n_directions: int = 5) -> jnp.ndarray:
        """
        Return top eigenvectors (stiff/well-constrained directions).
        
        These are the directions in parameter space where the data
        provides strong constraints. Changes along these directions
        significantly affect the model predictions.
        
        Args:
            n_directions: Number of top eigenvectors to return
            
        Returns:
            Matrix of shape (n_directions, n_params) - each row is an eigenvector
            
        Complexity: O(n²) for symmetric eigendecomposition
        """
        eigvals, eigvecs = jnp.linalg.eigh(self.matrix)
        # Sort by eigenvalue descending
        idx = jnp.argsort(eigvals)[::-1]
        return eigvecs[:, idx[:n_directions]].T


class FisherAtom:
    """
    Helper class for computing Fisher Information in neural networks.
    
    Provides utilities for:
    - Layer-wise Fisher computation
    - Kronecker-factored approximations (K-FAC style)
    - Empirical Fisher from mini-batches
    """
    
    @staticmethod
    def empirical_fisher(loss_fn: Callable,
                         params: List[jnp.ndarray],
                         inputs: jnp.ndarray,
                         targets: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Compute empirical Fisher Information for neural network parameters.
        
        The empirical Fisher uses the model's own predictions rather than
        the true data distribution:
            F_emp = E_{x~data}[∇log p(y|x,θ) ∇log p(y|x,θ)^T]
        
        This is what's typically used in practice (e.g., in K-FAC, Adam).
        
        Args:
            loss_fn: Function (params, inputs, targets) -> scalar loss
            params: List of parameter arrays
            inputs: Input data batch
            targets: Target labels/values
            
        Returns:
            List of Fisher matrices, one per parameter
        """
        def single_sample_loss(p, x, y):
            return loss_fn(p, x[None], y[None])
        
        # Compute per-sample gradients
        grad_fn = jax.grad(single_sample_loss)
        
        fisher_matrices = []
        n_samples = inputs.shape[0]
        
        for param_idx in range(len(params)):
            grads = []
            for i in range(n_samples):
                g = grad_fn(params, inputs[i], targets[i])
                grads.append(g[param_idx].flatten())
            
            grads = jnp.stack(grads)
            # Fisher = E[g g^T]
            fisher = jnp.mean(
                jax.vmap(lambda g: jnp.outer(g, g))(grads),
                axis=0
            )
            fisher_matrices.append(fisher)
        
        return fisher_matrices
    
    @staticmethod
    def diagonal_fisher(loss_fn: Callable,
                        params: List[jnp.ndarray],
                        inputs: jnp.ndarray,
                        targets: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Compute diagonal approximation to Fisher Information.
        
        Much cheaper than full Fisher: O(n) vs O(n²) storage.
        Used in Adam-style optimizers.
        
        Returns:
            List of diagonal Fisher vectors (not matrices)
        """
        def single_sample_loss(p, x, y):
            return loss_fn(p, x[None], y[None])
        
        grad_fn = jax.grad(single_sample_loss)
        
        diagonal_fishers = []
        n_samples = inputs.shape[0]
        
        for param_idx in range(len(params)):
            grad_sq_sum = jnp.zeros_like(params[param_idx])
            
            for i in range(n_samples):
                g = grad_fn(params, inputs[i], targets[i])
                grad_sq_sum += g[param_idx] ** 2
            
            diagonal_fishers.append(grad_sq_sum / n_samples)
        
        return diagonal_fishers


# =============================================================================
# Common Distribution Fisher Matrices (Analytical)
# =============================================================================

def fisher_gaussian(mu: jnp.ndarray, sigma: jnp.ndarray) -> FisherMetric:
    """
    Fisher Information for multivariate Gaussian N(μ, Σ).
    
    For the mean parameters μ:
        F_μ = Σ^{-1}
    
    For the full parameterization (μ, Σ), the Fisher metric is block diagonal
    with more complex structure for the covariance part.
    
    Args:
        mu: Mean vector
        sigma: Covariance matrix
        
    Returns:
        FisherMetric for mean parameters
    """
    fisher_matrix = jnp.linalg.inv(sigma)
    return FisherMetric(fisher_matrix, params=mu)


def fisher_categorical(probs: jnp.ndarray) -> FisherMetric:
    """
    Fisher Information for categorical distribution.
    
    For probabilities p = (p_1, ..., p_K):
        F_ij = δ_ij / p_i  (diagonal)
    
    Args:
        probs: Probability vector (sums to 1)
        
    Returns:
        FisherMetric for probability simplex
    """
    # Avoid division by zero
    probs_safe = jnp.maximum(probs, 1e-10)
    fisher_matrix = jnp.diag(1.0 / probs_safe)
    return FisherMetric(fisher_matrix, params=probs)


def fisher_exponential_family(natural_params: jnp.ndarray,
                              sufficient_stats_cov: jnp.ndarray) -> FisherMetric:
    """
    Fisher Information for exponential family in natural parameters.
    
    For exponential family p(x|η) = h(x) exp(η·T(x) - A(η)):
        F = Cov[T(x)] = ∇²A(η)
    
    The Fisher metric equals the covariance of sufficient statistics,
    which equals the Hessian of the log-partition function.
    
    Args:
        natural_params: Natural parameters η
        sufficient_stats_cov: Covariance of sufficient statistics
        
    Returns:
        FisherMetric
    """
    return FisherMetric(sufficient_stats_cov, params=natural_params)


__all__ = [
    'FisherMetric',
    'FisherAtom',
    'fisher_gaussian',
    'fisher_categorical',
    'fisher_exponential_family',
]

