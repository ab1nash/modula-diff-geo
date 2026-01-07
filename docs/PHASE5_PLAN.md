# Phase 5: Information Geometry Integration

## Overview

Phase 5 extends the geometric covariance framework with Information Geometry constructs, providing:
1. **Fisher Information Metric** - Natural Riemannian metric on statistical manifolds
2. **SPD Manifold Operations** - Symmetric Positive Definite matrix geometry
3. **Bregman Divergence** - Asymmetric divergence generalizing KL-divergence

These components enable geometry-aware learning on probability distributions and covariance matrices, with applications in:
- Neural network natural gradient descent
- EEG/BCI signal covariance classification
- Bayesian inference with geometric priors

---

## 5.1 FisherMetric

### Description
The Fisher Information Matrix defines a Riemannian metric on the space of probability distributions. For a parametric family p(x|θ), the metric is:

```
g_ij(θ) = E_p[∂_i log p(x|θ) · ∂_j log p(x|θ)]
```

This metric captures the local geometry of the statistical manifold.

### Implementation Location
`geometric/information.py`

### Class Structure
```python
class FisherMetric(MetricTensor):
    """
    Fisher Information as a Riemannian metric on statistical manifolds.
    
    Supports:
    - Automatic metric computation from log-likelihood gradients
    - Natural gradient computation: ∇_nat L = F^{-1} ∇L
    - Cramér-Rao bound verification
    """
    
    def __init__(self, log_likelihood_fn: Callable, params: jnp.ndarray):
        """
        Args:
            log_likelihood_fn: Function computing log p(x|θ)
            params: Current parameter values θ
        """
        self.log_likelihood_fn = log_likelihood_fn
        self.params = params
        super().__init__(self._compute_fisher_matrix())
    
    def _compute_fisher_matrix(self) -> jnp.ndarray:
        """Compute F_ij = E[∂_i log p · ∂_j log p] via sampling."""
        pass
    
    def natural_gradient(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """Compute F^{-1} ∇L for natural gradient descent."""
        return self.raise_index(gradient)
    
    @property
    def cramér_rao_bound(self) -> jnp.ndarray:
        """Return F^{-1} as lower bound on estimator variance."""
        return jnp.linalg.inv(self.matrix)
```

### Mathematical Invariants (Test Suite 5.1)

| Test ID | Property | Mathematical Statement | Verification |
|---------|----------|----------------------|--------------|
| T5.1.1 | Positive definiteness | g_ij(θ) > 0 | Check eigenvalues > 0 |
| T5.1.2 | Sufficient statistic invariance | Metric unchanged under sufficient statistics | Transform and compare |
| T5.1.3 | Cramér-Rao bound | Var(θ̂) ≥ F^{-1} | Verify bound in estimation |
| T5.1.4 | Natural gradient optimality | Fisher-Rao gradient is steepest in KL | Compare convergence rates |
| T5.1.5 | Reparameterization covariance | g'_ij = (∂θ/∂θ')^T g (∂θ/∂θ') | Transform parameters |

### Dependencies
- `geometric.metric.MetricTensor` (base class)
- JAX autodiff for gradient computation
- Optional: Monte Carlo sampling utilities

---

## 5.2 SPDManifold

### Description
The manifold of Symmetric Positive Definite (SPD) matrices with the affine-invariant Riemannian metric. This manifold appears in:
- Covariance matrix estimation (EEG, fMRI)
- Diffusion tensor imaging
- Kernel methods (Gram matrices)

The geodesic distance is:
```
d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
```

### Implementation Location
`geometric/spd.py`

### Class Structure
```python
class SPDManifold:
    """
    Manifold of Symmetric Positive Definite matrices.
    
    Provides:
    - Affine-invariant metric: d(WAW^T, WBW^T) = d(A, B)
    - Geodesic computation (matrix logarithm/exponential)
    - Fréchet mean (geometric mean)
    - Tangent space operations for learning
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Matrix dimension (n×n SPD matrices)
        """
        self.dim = dim
    
    def is_spd(self, A: jnp.ndarray) -> bool:
        """Check if matrix is SPD."""
        return jnp.all(jnp.linalg.eigvalsh(A) > 0)
    
    def distance(self, A: jnp.ndarray, B: jnp.ndarray) -> float:
        """Compute affine-invariant Riemannian distance."""
        pass
    
    def geodesic(self, A: jnp.ndarray, B: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute point along geodesic: γ(t) from A to B."""
        pass
    
    def frechet_mean(self, matrices: List[jnp.ndarray], 
                     weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute weighted Fréchet mean (geometric center)."""
        pass
    
    def log_map(self, A: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        """Logarithmic map: SPD → Tangent space at P."""
        pass
    
    def exp_map(self, V: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        """Exponential map: Tangent space at P → SPD."""
        pass


class SPDAtom(GeometricAtom):
    """
    Geometric atom operating on SPD manifold.
    
    For neural networks that process covariance matrices
    (e.g., EEG classification, graph Laplacians).
    """
    pass
```

### Mathematical Invariants (Test Suite 5.2)

| Test ID | Property | Mathematical Statement | Verification |
|---------|----------|----------------------|--------------|
| T5.2.1 | Affine invariance | d(WAWᵀ, WBWᵀ) = d(A, B) | Apply random W, verify |
| T5.2.2 | Geodesic stays in SPD | γ(t) ∈ SPD for t ∈ [0,1] | Check positive definiteness |
| T5.2.3 | No swelling effect | det(geometric mean) ≤ max(det(Aᵢ)) | Compare determinants |
| T5.2.4 | Fréchet mean convergence | Iterative mean converges | Track convergence |
| T5.2.5 | Log/Exp map inverses | exp_P(log_P(Q)) = Q | Round-trip verification |
| T5.2.6 | Triangle inequality | d(A,C) ≤ d(A,B) + d(B,C) | Verify metric property |

### Key Algorithms

**Geodesic computation:**
```
γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}
```

**Fréchet mean (iterative):**
```
repeat:
    M_new = M^{1/2} exp(∑ w_i log(M^{-1/2} A_i M^{-1/2})) M^{1/2}
until convergence
```

### Dependencies
- `scipy.linalg.sqrtm` or JAX equivalent for matrix square root
- `scipy.linalg.logm` / `scipy.linalg.expm` for matrix log/exp
- Eigendecomposition utilities

---

## 5.3 BregmanDivergence

### Description
Bregman divergences are a family of asymmetric "distances" generated by convex functions. For a strictly convex function φ:

```
D_φ(p || q) = φ(p) - φ(q) - ⟨∇φ(q), p - q⟩
```

Important special cases:
- **KL-divergence**: φ(p) = p log p (negative entropy)
- **Squared Euclidean**: φ(p) = ||p||²
- **Itakura-Saito**: φ(p) = -log(p)

### Implementation Location
`geometric/divergence.py`

### Class Structure
```python
class BregmanDivergence:
    """
    Bregman divergence from a convex generating function.
    
    Properties:
    - Non-symmetric: D(p||q) ≠ D(q||p) in general
    - Non-negative: D(p||q) ≥ 0, = 0 iff p = q
    - Convex in first argument
    
    The dual structure induces two affine connections (e/m connections)
    that are the foundation of information geometry.
    """
    
    def __init__(self, phi: Callable[[jnp.ndarray], float],
                 grad_phi: Optional[Callable] = None):
        """
        Args:
            phi: Convex generating function
            grad_phi: Gradient of phi (computed via autodiff if not provided)
        """
        self.phi = phi
        self.grad_phi = grad_phi or jax.grad(phi)
    
    def __call__(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """Compute D_φ(p || q)."""
        return (self.phi(p) - self.phi(q) 
                - jnp.dot(self.grad_phi(q), p - q))
    
    def dual_divergence(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """Compute D_φ*(q || p) using Legendre dual."""
        pass
    
    def e_projection(self, p: jnp.ndarray, 
                     constraint_set: Callable) -> jnp.ndarray:
        """Project onto constraint using exponential connection."""
        pass
    
    def m_projection(self, p: jnp.ndarray,
                     constraint_set: Callable) -> jnp.ndarray:
        """Project onto constraint using mixture connection."""
        pass


# Common Bregman divergences
class KLDivergence(BregmanDivergence):
    """Kullback-Leibler divergence (φ = p log p)."""
    pass

class SquaredEuclidean(BregmanDivergence):
    """Squared Euclidean distance (φ = ||p||²)."""
    pass

class ItakuraSaito(BregmanDivergence):
    """Itakura-Saito divergence (φ = -log p)."""
    pass
```

### Mathematical Invariants (Test Suite 5.3)

| Test ID | Property | Mathematical Statement | Verification |
|---------|----------|----------------------|--------------|
| T5.3.1 | Non-symmetry | D(P‖Q) ≠ D(Q‖P) in general | Compute both directions |
| T5.3.2 | Non-negativity | D(P‖Q) ≥ 0, = 0 iff P = Q | Test sample distributions |
| T5.3.3 | Convexity | D convex in first argument | Check second derivative |
| T5.3.4 | EM algorithm geometry | e/m projections alternate | Visualize on simplex |
| T5.3.5 | Pythagorean theorem | D(p‖q) = D(p‖r) + D(r‖q) for e-geodesic | Verify for aligned points |
| T5.3.6 | Dual divergence | D_φ*(q‖p) = D_φ(p‖q) | Legendre transform test |

### Applications in Modula
- **Natural gradient**: Use KL divergence geometry for optimization
- **Variational inference**: e/m projections for ELBO optimization
- **Mixture models**: Geometry of categorical distributions

---

## Implementation Order

```
Week 1: Core Infrastructure
├── geometric/information.py (FisherMetric stub)
├── geometric/spd.py (SPDManifold core)
└── tests/test_phase5_spd.py (T5.2.x tests)

Week 2: Fisher Information
├── FisherMetric full implementation
├── Natural gradient integration
└── tests/test_phase5_fisher.py (T5.1.x tests)

Week 3: Bregman Divergences
├── geometric/divergence.py
├── Common divergences (KL, Euclidean, IS)
└── tests/test_phase5_bregman.py (T5.3.x tests)

Week 4: Integration & Applications
├── SPDAtom for covariance classification
├── FisherModule for natural gradient training
└── Integration tests with existing atoms
```

---

## Dependencies on Earlier Phases

| Component | Depends On |
|-----------|------------|
| FisherMetric | MetricTensor (Phase 1), dualize methods (Phase 2) |
| SPDManifold | MetricTensor, ParallelTransport (Phase 4) |
| BregmanDivergence | RandersMetric (Phase 2) for asymmetric geometry |
| SPDAtom | GeometricAtom (Phase 3) |

---

## Computational Considerations

### Performance Targets
| Operation | Target Complexity | Notes |
|-----------|------------------|-------|
| Fisher matrix | O(n² × samples) | Sampling-based estimate |
| SPD geodesic | O(n³) | Matrix sqrt/log |
| Fréchet mean | O(k × n³) iterations | k iterations to converge |
| Bregman divergence | O(n) | Single evaluation |

### Numerical Stability
- **Matrix logarithm**: Use eigendecomposition with safeguards for small eigenvalues
- **Fisher matrix**: Add regularization (ε·I) for invertibility
- **SPD projection**: Clip eigenvalues to ensure positivity

### JAX Optimization
```python
# JIT-compile geodesic computation
@jax.jit
def spd_geodesic(A, B, t):
    A_sqrt = matrix_sqrt(A)
    A_inv_sqrt = matrix_sqrt(jnp.linalg.inv(A))
    inner = matrix_power(A_inv_sqrt @ B @ A_inv_sqrt, t)
    return A_sqrt @ inner @ A_sqrt
```

---

## Test File Structure

```
tests/
├── test_phase5_fisher.py      # T5.1.x tests
├── test_phase5_spd.py         # T5.2.x tests  
├── test_phase5_bregman.py     # T5.3.x tests
└── test_phase5_integration.py # End-to-end tests
```

---

## Success Criteria

### Phase Gate
- All T5.x tests pass
- SPD operations verified on synthetic covariance data
- Fisher metric produces valid natural gradients
- Bregman divergences satisfy all axioms

### Proof of Concept
1. **EEG Classification**: SPDManifold correctly classifies synthetic EEG covariances
2. **Natural Gradient Training**: Fisher metric accelerates training on simple model
3. **EM Algorithm**: Bregman projections correctly implement EM steps

---

## References

1. Amari, S. (2016). *Information Geometry and Its Applications*
2. Pennec, X. et al. (2006). "A Riemannian Framework for Tensor Computing"
3. Banerjee, A. et al. (2005). "Clustering with Bregman Divergences"
4. Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method"

---

## Open Questions

1. **Fisher matrix estimation**: Use exact computation (small models) or sampling?
2. **SPD tangent space**: Which parameterization for learning (log-Euclidean vs affine)?
3. **Bregman integration**: How to compose with Riemannian/Finsler atoms?
4. **GPU efficiency**: Matrix log/exp operations - custom kernels needed?

