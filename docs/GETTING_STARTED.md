# Getting Started with DiffGeo

## Abstract

**DiffGeo** extends Modula with differential geometry primitives for covariant pattern mining. While Modula treats neural networks as structured mathematical objects with duality and projection operations, DiffGeo adds explicit geometric structure—enabling proper handling of tensor variance (vectors vs gradients), asymmetric metrics (Finsler geometry), manifold constraints (SPD, SO(3)), and information geometry (Fisher metrics). The core insight is that gradients are *covectors* living in dual space, not vectors—a distinction invisible in Euclidean space but critical for non-Euclidean data. DiffGeo provides: (1) a type system for tracking tensor variance and parity through compositions, (2) geometric neural network layers with proper gradient transformations, (3) manifold operations (SPD, Lie groups, statistical manifolds), (4) information geometry (Fisher metric, Bregman divergences, geometry extraction), and (5) optimizers that respect manifold structure. Benchmarks show 60-70% RMSE improvements on manifold-structured data (EEG covariance, motion capture) compared to Euclidean baselines.

---

## Quick Start (5-minute read)

### Installation

```bash
pip install -e ".[test]"
```

### Essential Directory Structure

```
diffgeo/
├── core/
│   └── types.py          # TensorVariance, Parity, GeometricSignature
├── geometry/
│   ├── metric.py         # MetricTensor, GeometricVector
│   ├── finsler.py        # RandersMetric, FinslerDualizer
│   ├── spd.py            # SPDManifold, SPDMetricTensor
│   └── lie_groups.py     # SO(3) exp/log, retractions
├── information/
│   ├── fisher.py         # FisherMetric, natural gradient
│   ├── divergence.py     # KL, Bregman, Alpha divergences
│   └── extractor.py      # DataGeometryExtractor
├── nn/
│   ├── atoms.py          # FinslerLinear, TwistedEmbed
│   ├── bonds.py          # MetricTransition, ParallelTransport
│   └── transformer.py    # GeometricGPT, GeometricAttention
├── optim/
│   └── optimizer.py      # GeometricOptimizer, NaturalGradientOptimizer
└── cli.py                # Command-line interface
```

### Key Components at a Glance

| Component | File | Purpose |
|-----------|------|---------|
| **Type System** | `core/types.py` | Track contravariant/covariant, parity |
| **Finsler Metrics** | `geometry/finsler.py` | Asymmetric distances: F(v) ≠ F(-v) |
| **SPD Manifold** | `geometry/spd.py` | Covariance matrices, geodesics, Fréchet mean |
| **Lie Groups** | `geometry/lie_groups.py` | O(1) SO(3) operations via Rodrigues |
| **Fisher Metric** | `information/fisher.py` | Natural gradient, sloppy model analysis |
| **FinslerLinear** | `nn/atoms.py` | Linear layer with asymmetric weight updates |
| **TwistedEmbed** | `nn/atoms.py` | Orientation-sensitive embedding |
| **Geometric Optimizer** | `optim/optimizer.py` | Manifold-respecting gradient descent |

### Minimal Example

```python
from diffgeo import FinslerLinear, SPDManifold, TwistedEmbed
import jax

key = jax.random.PRNGKey(0)

# 1. Asymmetric linear layer (directed data)
finsler = FinslerLinear(fanout=64, fanin=32, drift_strength=0.3)
weights = finsler.initialize(key)

# 2. SPD manifold (covariance data)
spd = SPDManifold(dim=8)
P, Q = ...  # Your covariance matrices
dist = spd.distance(P, Q)
midpoint = spd.geodesic(P, Q, t=0.5)  # Stays in SPD cone!

# 3. Orientation-sensitive embedding (chiral data)
embed = TwistedEmbed(dEmbed=32, numEmbed=1000)
left = embed.forward(indices, weights, orientation=-1.0)
right = embed.forward(indices, weights, orientation=+1.0)
```

### CLI

```bash
diffgeo info              # Show all components
diffgeo demo spd          # SPD manifold demo
diffgeo demo finsler      # Asymmetric metrics demo
diffgeo benchmark         # Performance comparison
```

### Example Notebooks

- [`diffgeo-tutorial.ipynb`](../examples/diffgeo-tutorial.ipynb) — Introduction to geometric concepts
- [`manifold-visualizer-demo.ipynb`](../examples/manifold-visualizer-demo.ipynb) — Visualizing manifold operations

---

## Comprehensive Guide (15-minute read)

### 1. The Core Problem: Metric Blindness

In standard deep learning, we write:

```python
weights = weights - lr * gradient
```

This is **geometrically incorrect**. Parameters are *vectors* (contravariant), but gradients are *covectors* (covariant)—they live in different spaces. Subtracting them only makes sense through a metric:

```python
update = metric.raise_index(gradient)  # covector → vector
weights = weights - lr * update
```

In Euclidean space (identity metric), vectors = covectors, so we don't notice. On curved manifolds (covariance matrices, rotations, probability distributions), ignoring this distinction causes coordinate-dependent artifacts.

**DiffGeo makes the metric explicit**, enabling proper transformations for non-Euclidean data.

### 2. Core Type System (`diffgeo.core`)

The foundation is a type taxonomy from differential geometry:

```python
from diffgeo import TensorVariance, Parity, MetricType, GeometricSignature
```

**TensorVariance** — How tensors transform under basis change:
- `CONTRAVARIANT`: Vectors, velocities, displacements (transforms with J⁻¹)
- `COVARIANT`: Gradients, 1-forms (transforms with J)
- `MIXED`: Operators with both upper and lower indices
- `SCALAR`: Invariant quantities

**Parity** — Behavior under reflection:
- `EVEN (+1)`: True tensors (e.g., electric field)
- `ODD (-1)`: Pseudotensors (e.g., magnetic field, chirality)

**GeometricSignature** — Complete module metadata:
```python
signature = GeometricSignature(
    domain=TensorVariance.CONTRAVARIANT,
    codomain=TensorVariance.CONTRAVARIANT,
    parity=Parity.EVEN,
    metric_type=MetricType.FINSLER,
    dim_in=64,
    dim_out=128
)
```

The signature enables **type-safe composition**—incompatible modules (e.g., outputting vectors into a gradient-expecting layer) raise errors at construction time.

### 3. Geometry Module (`diffgeo.geometry`)

#### 3.1 Riemannian Metrics (`metric.py`)

`MetricTensor` provides index raising/lowering—the core operation converting between vectors and covectors:

```python
from diffgeo import MetricTensor

metric = MetricTensor(matrix)  # SPD matrix defining the metric
vector = metric.raise_index(covector)    # ♯ (sharp) - covector → vector
covector = metric.lower_index(vector)    # ♭ (flat) - vector → covector
```

#### 3.2 Finsler Geometry (`finsler.py`)

Finsler metrics generalize Riemannian metrics to allow **asymmetry**: the cost to travel A→B may differ from B→A.

**RandersMetric** is the most tractable Finsler structure:

```
F(v) = √(vᵀAv) + bᵀv
```

where A is the Riemannian part and b is the "drift" (wind, current, bias):

```python
from diffgeo import RandersMetric

randers = RandersMetric(A=covariance, b=drift_vector)

# Costs are asymmetric!
cost_forward = randers.norm(v)     # ~1.4 (with drift)
cost_backward = randers.norm(-v)   # ~0.6 (against drift)

# Finsler dualization
from diffgeo import FinslerDualizer
dualizer = FinslerDualizer(randers)
update = dualizer.dualize(gradient)
```

**Applications**: Directed graphs, causal modeling, thermodynamic flows, time series with directional bias.

#### 3.3 SPD Manifold (`spd.py`)

Symmetric Positive Definite matrices (covariance matrices, diffusion tensors) form a curved manifold with negative curvature. Standard Euclidean operations fail:
- Euclidean interpolation can leave the SPD cone
- Euclidean mean has "swelling effect"

```python
from diffgeo import SPDManifold

spd = SPDManifold(dim=8)

# Geodesic distance (affine-invariant)
dist = spd.distance(P, Q)

# Geodesic interpolation (stays in SPD cone)
midpoint = spd.geodesic(P, Q, t=0.5)

# Fréchet mean (geometric center)
mean = spd.frechet_mean([P1, P2, P3])

# Tangent space projection (for classification)
tangent_vectors = spd.tangent_space_projection(matrices, reference=mean)
```

**Applications**: EEG/BCI classification, diffusion tensor imaging, radar signal processing.

#### 3.4 Lie Groups (`lie_groups.py`)

Efficient operations on rotation groups using closed-form solutions:

```python
from diffgeo.geometry import so3_exp, so3_log, so3_geodesic

# Rodrigues formula: O(1) instead of O(n³)
R = so3_exp(omega)  # Lie algebra → Lie group
omega = so3_log(R)  # Lie group → Lie algebra

# Geodesic interpolation on SO(3)
R_interp = so3_geodesic(R1, R2, t=0.5)
```

**Retractions** for optimization on manifolds:
- `qr_retraction`: O(n²) for Stiefel/orthogonal manifolds
- `polar_retraction`: Closest orthogonal matrix
- `cayley_retraction`: For SO(n)
- `spd_retraction`: Project to SPD cone

### 4. Information Geometry (`diffgeo.information`)

#### 4.1 Fisher Metric (`fisher.py`)

The Fisher Information Matrix is the natural Riemannian metric on parameter spaces:

```python
from diffgeo import FisherMetric

# From log-likelihood function
fisher = FisherMetric.from_log_likelihood(log_prob_fn, params, samples)

# Natural gradient (invariant to reparameterization)
natural_grad = fisher.natural_gradient(euclidean_grad)

# Sloppy model analysis
effective_dim = fisher.effective_dimension()
condition = fisher.condition_number()
stiff_dirs = fisher.stiff_directions(n=5)
```

**Diagonal approximation** for O(n) instead of O(n³):
```python
diag_natural_grad = fisher.natural_gradient_diagonal(grad)
```

#### 4.2 Divergences (`divergence.py`)

Bregman divergences generalize KL and provide dual coordinate structure:

```python
from diffgeo import KLDivergence, SquaredEuclidean, ItakuraSaito, LogDet

kl = KLDivergence()
div = kl(p, q)  # KL(p || q)

# Bregman-Fisher connection
from diffgeo import fisher_from_bregman, DuallyFlatManifold
fisher_matrix = fisher_from_bregman(bregman, point)
```

#### 4.3 Geometry Extraction (`extractor.py`)

Automatically learn geometry from raw data:

```python
from diffgeo import DataGeometryExtractor, SPDGeometryExtractor

extractor = DataGeometryExtractor()

# From time series → covariance manifold
manifold = extractor.from_time_series(eeg_data, window_size=64)

# From pairwise distances → Finsler metric
embedding, metric = extractor.from_pairwise_distances(distances)
```

#### 4.4 Statistical Manifold (`manifolds.py`)

Any parametric model defines a statistical manifold with Fisher geometry:

```python
from diffgeo import StatisticalManifold

manifold = StatisticalManifold.from_log_prob(log_prob_fn, params, samples)
natural_grad = manifold.natural_gradient(euclidean_grad)
dist = manifold.geodesic_distance(other_params)

# Finsler extension for asymmetric data
if manifold.is_asymmetric():
    randers = manifold.as_randers_metric()
```

### 5. Neural Network Layers (`diffgeo.nn`)

#### 5.1 Geometric Atoms (`atoms.py`)

**FinslerLinear** — Linear layer with asymmetric weight updates:
```python
from diffgeo import FinslerLinear

layer = FinslerLinear(fanout=128, fanin=64, drift_strength=0.3)
weights = layer.initialize(key)  # Returns [W, drift]

y = layer.forward(x, weights)
dualized = layer.dualize(gradients)  # Finsler-aware orthogonalization
```

**TwistedEmbed** — Orientation-sensitive embedding (parity = -1):
```python
from diffgeo import TwistedEmbed

embed = TwistedEmbed(dEmbed=64, numEmbed=1000)
right = embed.forward(indices, weights, orientation=+1.0)
left = embed.forward(indices, weights, orientation=-1.0)
# right ≠ left (distinguishes chirality)
```

**ContactAtom** — Projects onto constraint hypersurface (conservation laws).

#### 5.2 Geometric Bonds (`bonds.py`)

Bonds handle transitions between different geometric structures:

- `MetricTransition`: Smooth interpolation between metric types
- `ParallelTransport`: Transport vectors between tangent spaces
- `SymplecticBond`: Preserve Hamiltonian structure
- `RopeJIT`: JIT-compatible rotary position embedding

#### 5.3 Geometric Transformer (`transformer.py`)

```python
from diffgeo import GeometricGPT, create_geometric_gpt

model = create_geometric_gpt(
    vocab_size=65,
    num_heads=4,
    d_embed=128,
    num_blocks=4,
    drift_strength=0.3,
    orientation=+1.0
)

# For chirality-invariant learning
from diffgeo import create_chiral_pair
left_model, right_model = create_chiral_pair(vocab_size=65)
```

### 6. Optimizers (`diffgeo.optim`)

**GeometricOptimizer** — Respects covariant/contravariant distinction:

```python
from diffgeo import GeometricOptimizer, NaturalGradientOptimizer

# With statistical manifold
optimizer = GeometricOptimizer(manifold, learning_rate=0.01)
state = optimizer.init(params)
state = optimizer.step(state, gradient)

# With manifold constraints (retraction)
state = optimizer.step_with_retraction(state, gradient, manifold_type="spd")
```

**NaturalGradientOptimizer** — Classic natural gradient with Fisher metric.

**FinslerOptimizer** — For asymmetric manifolds with drift.

### 7. Computational Efficiency

| Operation | Naïve | DiffGeo | Method |
|-----------|-------|---------|--------|
| SO(3) exp/log | O(n³) eigendecomp | **O(1)** | Rodrigues formula |
| Natural gradient | O(n³) full inverse | **O(n)** | Diagonal Fisher |
| Parallel transport | O(n²) | **O(n)** | First-order projection |
| Retractions | O(n³) exp | **O(n²)** | QR/Polar/Cayley |

### 8. Testing

```bash
# All tests (280+ tests)
pytest tests/ -v

# By phase
pytest tests/ -m phase1 -v    # Core type system
pytest tests/ -m phase2 -v    # Dualization & Finsler
pytest tests/ -m phase5 -v    # Information geometry

# Real-world hypothesis tests
pytest tests/realworld/ -v

# Mathematical invariants
pytest tests/ -m invariant -v
```

### 9. Benchmarks

Run benchmarks comparing geometric vs Euclidean methods:

```bash
python run_fisher_benchmarks.py --quick    # Fast test
python run_fisher_benchmarks.py            # Standard
python run_fisher_benchmarks.py --full     # Full (3 runs, 3000 epochs)
```

Results on manifold-structured data:
- **PhysioNet EEG (SPD)**: 70% RMSE improvement
- **CMU MoCap (SO(3))**: 64% RMSE improvement
- **GHCN Climate (Spherical)**: 18% RMSE improvement

### 10. Relation to Modula

DiffGeo builds on Modula's core abstractions:

| Modula | DiffGeo Extension |
|--------|-------------------|
| `Linear` | `FinslerLinear` (asymmetric metric) |
| `Embed` | `TwistedEmbed` (parity tracking) |
| `Bond` | `MetricTransition`, `ParallelTransport` |
| `dualize()` | Geometric dualization with metric |
| `project()` | Manifold retraction |

**What's Preserved**: Modular composition (@ operator), mass/sensitivity, tare(), JIT compatibility.

**What's Added**: Explicit geometric signatures, non-Euclidean metrics, manifold constraints.

---

## Further Reading

- **Mathematical Background**: [`diffgeo-tutorial.ipynb`](../examples/diffgeo-tutorial.ipynb)
- **API Reference**: [`docs/source/`](source/)
- **Design Documents**: [`ignore-docs/`](../ignore-docs/)
- **Phase 5 Plan**: [`PHASE5_PLAN.md`](PHASE5_PLAN.md)

