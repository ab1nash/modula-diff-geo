<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/modula.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/modula_light.svg">
  <img alt="modula logo" src="assets/modula.svg">
</picture>

[![Test Bed + unit tests](https://github.com/ab1nash/modula-diff-geo/actions/workflows/tests.yml/badge.svg)](https://github.com/ab1nash/modula-diff-geo/actions/workflows/tests.yml)

## 📊 Benchmark Highlights (2026-01-08)

Geometric methods show **significant improvements** on manifold-structured data:

| Dataset | Data Type | Modula (Baseline) | Geometric Method | RMSE Improvement |
|---------|-----------|-------------------|------------------|------------------|
| **PhysioNet EEG** | SPD Covariance | 0.447 ± 0.015 | **SPD Tangent: 0.132 ± 0.003** | **70.4% ↓** |
| **PhysioNet EEG** | SPD Covariance | 0.447 ± 0.015 | SPD Fisher: 0.133 ± 0.001 | 70.2% ↓ |
| **GHCN Climate** | Spherical + Values | 25.65 ± 2.48 | **Extracted Fisher: 21.00 ± 1.27** | **18.1% ↓** |
| **CMU MoCap** | SO(3) Joint Angles | 0.381 ± 0.005 | Extracted Fisher: 0.378 ± 0.004 | 0.9% ↓ |

**Key findings:**
- 🎯 **[SPD](https://en.wikipedia.org/wiki/Definite_matrix) data benefits most** from geometric methods (70%+ improvement on EEG covariance matrices)
- 🌍 **Fisher geometry discovers structure** automatically on spherical/mixed data
- 📐 **[MIS](docs/manifold_integrity_score.md) near zero** for geometric methods = predictions stay on the manifold

> *SPD = Symmetric Positive Definite matrices. See [Arsigny et al. (2006)](https://hal.inria.fr/inria-00071383/document) for Log-Euclidean metrics on SPD manifolds.*
>
> *Benchmark: 3 runs per condition, full training (3000 epochs, early stopping @ 150 patience)*

<details>
<summary><b>📋 Reproduce these results</b></summary>

```bash
# Install dependencies
pip install -e ".[test]"

# Quick test (fast, fewer epochs)
python run_fisher_benchmarks.py --quick

# Standard benchmark
python run_fisher_benchmarks.py

# Full benchmark (3 runs, 3000 epochs - used for table above)
python run_fisher_benchmarks.py --full

# Run specific datasets
python run_fisher_benchmarks.py --full --datasets physionet_eeg
python run_fisher_benchmarks.py --full --datasets ghcn_daily cmu_mocap

# Results saved to:
#   results/json/     - JSON with all metrics
#   results/figures/  - PNG visualizations
```

</details>

Modula is a deep learning library and a deep learning theory built hand-in-hand. Modula disentangles complex neural networks and turns them into structured mathematical objects called modules. This makes training faster and easier to scale, while also providing tools for understanding the properties of the trained network. Modula is built on top of [JAX](https://github.com/google/jax). More information is available in the [Modula docs](https://docs.modula.systems).

# Installation

Modula can be installed using pip:

```bash
pip install git+https://github.com/modula-systems/modula.git
```

Or you can clone the repository and install locally:

```bash
git clone https://github.com/modula-systems/modula.git
cd modula
pip install -e .
```

# Functionality

Modula provides a set of architecture-specific helper functions that are automatically constructed along with the network architecture itself. As an example, let's build a multi-layer perceptron:

```python
from modula.atom import Linear
from modula.bond import ReLU

mlp = Linear(10, 256)
mlp @= ReLU()
mlp @= Linear(256, 256)
mlp @= ReLU()
mlp @= Linear(256, 784)

mlp.jit() # makes everything run faster
```

Behind the scenes, Modula builds a function to randomly initialize the weights of the network:

```python
import jax

key = jax.random.PRNGKey(0)
weights = mlp.initialize(key)
```

Supposing we have used JAX to compute the gradient of our loss and stored this as `grad`, then we can use Modula to dualize the gradient, thereby accelerating our gradient descent training:

```python
dualized_grad = mlp.dualize(grad)
weights = [w - 0.1 * dg for w, dg in zip(weights, dualized_grad)]
```

And after the weight update, we can project the weights back to their natural constraint set:

```python
weights = mlp.project(weights)
```

In short, Modula lets us think about the weight space of our neural network as a somewhat classical optimization space, complete with duality and projection operations.

---

# DiffGeo: Differential Geometry Extensions

This branch extends Modula with **DiffGeo** (`diffgeo`) - differential geometry primitives for covariant pattern mining. DiffGeo equips neural network components with explicit geometric structure for handling complex data relationships.

## Quick Start

```bash
# Install with all dependencies
pip install -e ".[test]"

# Explore with CLI
diffgeo info              # Show all components
diffgeo demo spd          # SPD manifold demo
diffgeo demo finsler      # Asymmetric metrics demo
diffgeo demo chiral       # Chirality detection demo
diffgeo benchmark         # Performance comparison
```

## Core Concepts

| Concept | Description | Use Case |
|---------|-------------|----------|
| **Tensor Variance** | Contravariant (vectors) vs covariant (gradients) | Proper gradient transformations |
| **Twisted Forms** | Orientation-sensitive tensors (parity=-1) | Chiral molecules, handedness |
| **Finsler Metrics** | Asymmetric norms: F(v) ≠ F(-v) | Causality, directed graphs |
| **SPD Manifold** | Symmetric Positive Definite matrices | Covariance, EEG/BCI data |

## Examples

### 1. Geometric Linear Layer

```python
from diffgeo import GeometricLinear, Parity
import jax

key = jax.random.PRNGKey(0)

# Standard geometric layer (preserves vector type)
layer = GeometricLinear(fanout=128, fanin=64)
weights = layer.initialize(key)

# Forward pass
x = jax.random.normal(key, (64,))
y = layer.forward(x, weights)  # Shape: (128,)

# Geometric dualization (spectral normalization)
grad = jax.random.normal(key, (128, 64))
dual_grad = layer.dualize([grad])
```

### 2. Finsler Layer for Directed Data

For data with asymmetric relationships (causality, information flow):

```python
from diffgeo import FinslerLinear, RandersMetric
import jax.numpy as jnp

# FinslerLinear has a "drift" that makes certain directions cheaper
finsler = FinslerLinear(fanout=64, fanin=64, drift_strength=0.4)
weights = finsler.initialize(key)

# The drift introduces directional asymmetry
W, drift = weights[0], weights[1]
print(f"Drift direction: {drift[:4]}...")

# Randers metric: F(v) = sqrt(v^T A v) + b^T v
A = jnp.eye(64)
b = jnp.zeros(64).at[0].set(0.4)  # Drift in first dimension
randers = RandersMetric(A, b)

# Costs are different for opposite directions!
v = jnp.array([1.0] + [0.0]*63)
print(f"F(+v) = {randers.norm(v):.3f}")   # ~1.4 (with drift)
print(f"F(-v) = {randers.norm(-v):.3f}")  # ~0.6 (against drift)
```

### 3. SPD Manifold for Covariance Data

For data that lives on the cone of positive definite matrices (EEG, DTI):

```python
from diffgeo import SPDManifold
import jax.numpy as jnp

spd = SPDManifold(dim=8)

# Create SPD matrices (e.g., covariance matrices)
L1 = jax.random.normal(key, (8, 8))
P = L1 @ L1.T + 0.1 * jnp.eye(8)

L2 = jax.random.normal(jax.random.split(key)[0], (8, 8))
Q = L2 @ L2.T + 0.1 * jnp.eye(8)

# Riemannian distance (respects SPD geometry)
dist = spd.distance(P, Q)

# Geodesic interpolation (stays in SPD cone!)
midpoint = spd.geodesic(P, Q, t=0.5)
assert jnp.all(jnp.linalg.eigvalsh(midpoint) > 0)  # Still SPD!

# Fréchet mean (geometric average)
matrices = jnp.stack([P, Q])
mean = spd.frechet_mean(matrices)
```

### 4. Chirality Detection with Twisted Embed

For distinguishing mirror images (molecules, handedness):

```python
from diffgeo import TwistedEmbed

# TwistedEmbed is orientation-sensitive
embed = TwistedEmbed(dEmbed=32, numEmbed=1000)
weights = embed.initialize(key)

indices = jnp.array([42, 137, 256])

# Same indices, different chirality
right_handed = embed.forward(indices, weights, orientation=+1.0)
left_handed = embed.forward(indices, weights, orientation=-1.0)

# They're different! (but same magnitude)
print(f"||R - L|| = {jnp.linalg.norm(right_handed - left_handed):.4f}")
print(f"||R|| = ||L|| = {jnp.linalg.norm(right_handed):.4f}")
```

### 5. Missing Data with Geometric Imputation

```python
from tests.realworld.utils import DataMasker, MaskPattern, MissingDataEvaluator

# Mask some data
data = jax.random.normal(key, (100, 16))
masked = DataMasker.apply_mask(
    data, 
    missing_fraction=0.3,
    pattern=MaskPattern.UNIFORM_RANDOM,
    key=key
)

print(f"Missing: {masked.missing_fraction:.1%}")
print(f"Observed entries: {jnp.sum(masked.mask)}")

# Evaluate imputation quality with standard ML metrics
true_vals = data[~masked.mask]
pred_vals = jnp.zeros_like(true_vals)  # Zero imputation baseline

metrics = MissingDataEvaluator.compute_all_metrics(true_vals, pred_vals)
print(f"RMSE: {metrics.rmse:.4f}")
print(f"Hits@10: {metrics.hits_at_10:.2%}")
print(f"MRR: {metrics.mrr:.4f}")
```

## Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests (284 tests)
pytest tests/ -v

# Run by category
pytest tests/ -m phase1 -v           # Core type system
pytest tests/ -m phase2 -v           # Dualization & Finsler
pytest tests/ -m hypothesis -v       # Real-world hypothesis tests
pytest tests/realworld/ -v           # Missing data, SPD, chirality

# Mathematical invariant verification
pytest tests/ -m invariant -v

# Quick smoke test
pytest tests/realworld/test_utilities.py -v
```

## CLI Reference

```bash
diffgeo info              # Package info and components
diffgeo demo spd          # SPD manifold operations
diffgeo demo finsler      # Asymmetric Finsler metrics
diffgeo demo chiral       # Chirality (handedness) detection
diffgeo check invariants  # Run math invariant tests
diffgeo benchmark         # Forward pass performance
```

## Architecture

```
diffgeo/
├── core.py         # TensorVariance, Parity, MetricType, GeometricSignature
├── metric.py       # MetricTensor, GeometricVector
├── finsler.py      # RandersMetric, FinslerDualizer
├── module.py       # GeometricModule, GeometricAtom base classes
├── atoms.py        # GeometricLinear, FinslerLinear, TwistedEmbed, ContactAtom
├── bonds.py        # MetricTransition, ParallelTransport, SymplecticBond
├── spd.py          # SPDManifold, SPDMetricTensor, SPDClassifier
├── information.py  # FisherMetric, FisherAtom
├── divergence.py   # KLDivergence, BregmanDivergence, AlphaDivergence
└── cli.py          # Command-line interface
```

---

# References

Modula is based on two papers. The first is on the [modular norm](https://arxiv.org/abs/2405.14813):

```bibtex
@inproceedings{modular-norm,
  title     = {Scalable Optimization in the Modular Norm},
  author    = {Tim Large and Yang Liu and Minyoung Huh and Hyojin Bahng and Phillip Isola and Jeremy Bernstein},
  booktitle = {Neural Information Processing Systems},
  year      = {2024}
}
```

And the second is on [modular duality](https://arxiv.org/abs/2410.21265):

```bibtex
@article{modular-duality,
  title   = {Modular Duality in Deep Learning},
  author  = {Jeremy Bernstein and Laker Newhouse},
  journal = {arXiv:2410.21265},
  year    = {2024}
}
```

## Acknowledgements
We originally wrote Modula on top of PyTorch, but I ported the project over to JAX inspired by Jack Gallagher's [modulax](https://github.com/GallagherCommaJack/modulax).

## License
Modula is released under an [MIT license](/LICENSE).
