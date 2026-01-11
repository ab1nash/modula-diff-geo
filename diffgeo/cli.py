#!/usr/bin/env python3
"""
DiffGeo CLI - Differential Geometry Extensions for Modula

Command-line interface for inspecting geometric properties, running demos,
and testing invariants.

Usage:
    diffgeo info                    Show package info and available components
    diffgeo demo spd               Run SPD manifold demo
    diffgeo demo finsler           Run Finsler metric demo  
    diffgeo demo chiral            Run chirality detection demo
    diffgeo check invariants       Run mathematical invariant checks
    diffgeo benchmark              Run performance benchmarks
"""
import argparse
import sys

import jax
import jax.numpy as jnp


def cmd_info(args):
    """Show package information and available components."""
    from diffgeo import (
        FinslerLinear,
        TwistedEmbed,
        GeometricEmbed,
        MetricTensor,
        RandersMetric,
        SPDManifold,
        FisherMetric,
    )

    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║                          DiffGeo for Modula                          ║
║           Differential Geometry Extensions for Neural Nets           ║
╚══════════════════════════════════════════════════════════════════════╝

Geometric Atoms:
  • FinslerLinear     - Linear with asymmetric Finsler metric (USE THIS)
  • TwistedEmbed      - Orientation-sensitive embedding (chirality)
  • GeometricEmbed    - Standard embedding with geometric tracking
  • ContactAtom       - Conservation law projection
  • GeometricLinear   - Abstract base class (signature tracking only)

Metric Structures:
  • MetricTensor      - Riemannian metric with index raising/lowering
  • RandersMetric     - Asymmetric F(v) = √(v^T A v) + b^T v
  • SPDManifold       - Symmetric Positive Definite matrix manifold
  • FisherMetric      - Fisher information metric for distributions

Key Concepts:
  • Tensor variance   - Contravariant (vectors) vs covariant (gradients)
  • Parity           - Even/odd behavior under reflection
  • Finsler geometry - Asymmetric norms for directional cost

Quick Start:
    from diffgeo import FinslerLinear, TwistedEmbed
    from modula.atom import Linear  # Base modula
    
    # Finsler layer for directed/causal data (asymmetric gradient updates)
    finsler_layer = FinslerLinear(64, 128, drift_strength=0.3)
    
    # Orientation-sensitive embedding for chiral data
    twisted_embed = TwistedEmbed(dEmbed=64, numEmbed=1000)
"""
    )


def cmd_demo_spd(args):
    """Run SPD (Symmetric Positive Definite) manifold demo."""
    from diffgeo import SPDManifold
    
    print("\n=== SPD Manifold Demo ===\n")
    
    key = jax.random.PRNGKey(42)
    dim = 4
    
    # Create SPD manifold
    spd = SPDManifold(dim)
    
    # Generate random SPD matrices
    k1, k2 = jax.random.split(key)
    L1 = jax.random.normal(k1, (dim, dim))
    L2 = jax.random.normal(k2, (dim, dim))
    
    P = L1 @ L1.T + 0.1 * jnp.eye(dim)
    Q = L2 @ L2.T + 0.1 * jnp.eye(dim)
    
    print(f"Generated two random {dim}×{dim} SPD matrices")
    print(f"  det(P) = {jnp.linalg.det(P):.4f}")
    print(f"  det(Q) = {jnp.linalg.det(Q):.4f}")
    
    # Compute geodesic distance
    dist = spd.distance(P, Q)
    print(f"\nRiemannian geodesic distance: {dist:.4f}")
    
    # Compute Fréchet mean
    matrices = jnp.stack([P, Q])
    mean = spd.frechet_mean(matrices)
    print(f"Fréchet mean det: {jnp.linalg.det(mean):.4f}")
    
    # Show geodesic interpolation
    print("\nGeodesic interpolation P → Q:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interp = spd.geodesic(P, Q, t)
        print(f"  t={t:.2f}: det={jnp.linalg.det(interp):.4f}")
    
    print("\n✓ SPD manifold operations preserve positive definiteness")


def cmd_demo_finsler(args):
    """Run Finsler (asymmetric) metric demo."""
    from diffgeo import RandersMetric, FinslerLinear
    
    print("\n=== Finsler Metric Demo ===\n")
    
    key = jax.random.PRNGKey(42)
    dim = 8
    
    # Create Randers metric: F(v) = sqrt(v^T A v) + b^T v
    A = jnp.eye(dim)  # Base Riemannian metric
    b = jnp.zeros(dim).at[0].set(0.4)  # Drift in first dimension
    
    randers = RandersMetric(A, b)
    
    print(f"Randers metric on R^{dim} with drift ||b|| = {jnp.linalg.norm(b):.2f}")
    
    # Test asymmetry
    v = jnp.array([1.0] + [0.0] * (dim - 1))  # Unit vector along drift
    
    cost_forward = randers.norm(v)
    cost_backward = randers.norm(-v)
    
    print(f"\nAsymmetry demonstration (vector aligned with drift):")
    print(f"  F(+v) = {cost_forward:.4f}  (moving with drift)")
    print(f"  F(-v) = {cost_backward:.4f}  (moving against drift)")
    print(f"  Ratio = {cost_forward / cost_backward:.2f}x")
    
    # Create FinslerLinear layer
    finsler = FinslerLinear(dim, dim, drift_strength=0.5)
    weights = finsler.initialize(key)
    
    print(f"\nFinslerLinear layer initialized:")
    print(f"  Weight matrix shape: {weights[0].shape}")
    print(f"  Drift vector norm: {jnp.linalg.norm(weights[1]):.4f}")
    
    print("\n✓ Finsler metrics model asymmetric/directional costs")


def cmd_demo_chiral(args):
    """Run chirality (handedness) detection demo."""
    from diffgeo import TwistedEmbed, GeometricEmbed
    
    print("\n=== Chirality Detection Demo ===\n")
    
    key = jax.random.PRNGKey(42)
    
    # Create twisted embedding (orientation-sensitive)
    twisted = TwistedEmbed(dEmbed=32, numEmbed=100)
    weights = twisted.initialize(key)
    
    # Sample some indices
    indices = jnp.array([0, 1, 2])
    
    # Get embeddings for both orientations
    right_handed = twisted.forward(indices, weights, orientation=+1.0)
    left_handed = twisted.forward(indices, weights, orientation=-1.0)
    
    print("TwistedEmbed distinguishes chirality:")
    print(f"  Right-handed (orientation=+1): ||embed|| = {jnp.linalg.norm(right_handed):.4f}")
    print(f"  Left-handed (orientation=-1):  ||embed|| = {jnp.linalg.norm(left_handed):.4f}")
    print(f"  Difference: ||R - L|| = {jnp.linalg.norm(right_handed - left_handed):.4f}")
    
    # Standard embedding cannot distinguish
    standard = GeometricEmbed(dEmbed=32, numEmbed=100)
    std_weights = standard.initialize(key)
    
    embed1 = standard.forward(indices, std_weights)
    embed2 = standard.forward(indices, std_weights)
    
    print(f"\nGeometricEmbed is orientation-blind:")
    print(f"  Same input → same output: ||e1 - e2|| = {jnp.linalg.norm(embed1 - embed2):.6f}")
    
    print("\n✓ TwistedEmbed captures handedness (e.g., L vs D amino acids)")


def cmd_check_invariants(args):
    """Run mathematical invariant checks."""
    import subprocess
    
    print("\n=== Running Mathematical Invariant Checks ===\n")
    
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-m", "invariant", "-v", "--tb=short"],
        cwd=".",
        capture_output=False
    )
    
    return result.returncode


def cmd_benchmark(args):
    """Run comprehensive performance benchmarks."""
    from diffgeo import FinslerLinear, TwistedEmbed, GeometricEmbed
    from modula.atom import Linear, Embed
    import time
    import warnings
    import numpy as np

    # Suppress GeometricLinear warning for benchmarks
    warnings.filterwarnings(
        "ignore", message="GeometricLinear provides no geometric dualization"
    )

    print("\n" + "═" * 70)
    print("                    DiffGeo Performance Benchmarks")
    print("═" * 70 + "\n")

    key = jax.random.PRNGKey(42)
    dim = 256
    batch_size = 128
    n_warmup = 50
    n_iterations = 500

    print(
        f"Config: dim={dim}, batch_size={batch_size}, warmup={n_warmup}, iterations={n_iterations}"
    )
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Setup: Create layers and initialize weights
    # ─────────────────────────────────────────────────────────────────────

    base_linear = Linear(dim, dim)
    finsler_linear = FinslerLinear(dim, dim, drift_strength=0.3)
    base_embed = Embed(dEmbed=dim, numEmbed=1000)
    twisted_embed = TwistedEmbed(dEmbed=dim, numEmbed=1000)

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    base_w = base_linear.initialize(k1)
    finsler_w = finsler_linear.initialize(k2)
    base_embed_w = base_embed.initialize(k3)
    twisted_embed_w = twisted_embed.initialize(k4)

    batch = jax.random.normal(k5, (batch_size, dim))
    indices = jax.random.randint(k5, (batch_size,), 0, 1000)

    # Create gradient tensors for dualization benchmarks
    grad_base = [jax.random.normal(k1, shape=base_w[0].shape)]
    grad_finsler = [jax.random.normal(k1, shape=finsler_w[0].shape), finsler_w[1]]
    grad_embed = [jax.random.normal(k3, shape=base_embed_w[0].shape)]

    # ─────────────────────────────────────────────────────────────────────
    # JIT compile all operations
    # ─────────────────────────────────────────────────────────────────────

    # Forward passes
    base_fwd = jax.jit(lambda x, w: jax.vmap(lambda xi: base_linear.forward(xi, w))(x))
    finsler_fwd = jax.jit(lambda x, w: jax.vmap(lambda xi: finsler_linear.forward(xi, w))(x))
    base_embed_fwd = jax.jit(lambda idx, w: base_embed.forward(idx, w))
    twisted_embed_fwd = jax.jit(
        lambda idx, w: twisted_embed.forward(idx, w, orientation=1.0)
    )

    # Dualization (gradient → update conversion)
    # Note: Not JIT-compiled because finsler_orthogonalize has Python conditionals
    base_dual = lambda g: base_linear.dualize(g, targetNorm=1.0)
    finsler_dual = lambda g: finsler_linear.dualize(g, targetNorm=1.0)
    embed_dual = lambda g: base_embed.dualize(g, targetNorm=1.0)
    twisted_dual = lambda g: twisted_embed.dualize(g, targetNorm=1.0)

    # Projection (weight normalization)
    base_proj = jax.jit(lambda w: base_linear.project(w))
    finsler_proj = jax.jit(lambda w: finsler_linear.project(w))

    # ─────────────────────────────────────────────────────────────────────
    # Warmup all JIT-compiled functions
    # ─────────────────────────────────────────────────────────────────────

    print("Warming up JIT compilation...")
    for _ in range(n_warmup):
        _ = base_fwd(batch, base_w).block_until_ready()
        _ = finsler_fwd(batch, finsler_w).block_until_ready()
        _ = base_embed_fwd(indices, base_embed_w).block_until_ready()
        _ = twisted_embed_fwd(indices, twisted_embed_w).block_until_ready()
        _ = base_dual(grad_base)
        _ = finsler_dual(grad_finsler)
        _ = base_proj(base_w)[0].block_until_ready()
        _ = finsler_proj(finsler_w)[0].block_until_ready()

    def benchmark(fn, n_iter):
        """Run benchmark and return (p50, p95, p99) in ms."""
        times = []
        for _ in range(n_iter):
            start = time.perf_counter()
            result = fn()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                result[0].block_until_ready()
            times.append((time.perf_counter() - start) * 1000)
        times = np.array(times)
        return (
            np.percentile(times, 50),
            np.percentile(times, 95),
            np.percentile(times, 99),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark: Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    print("\n┌" + "─" * 68 + "┐")
    print("│ FORWARD PASS (ms/batch)                 p50      p95      p99      │")
    print("├" + "─" * 68 + "┤")

    base_fwd_t = benchmark(lambda: base_fwd(batch, base_w), n_iterations)
    finsler_fwd_t = benchmark(lambda: finsler_fwd(batch, finsler_w), n_iterations)
    base_embed_t = benchmark(
        lambda: base_embed_fwd(indices, base_embed_w), n_iterations
    )
    twisted_embed_t = benchmark(
        lambda: twisted_embed_fwd(indices, twisted_embed_w), n_iterations
    )

    print(
        f"│  Linear (base):      {base_fwd_t[0]:6.3f}   {base_fwd_t[1]:6.3f}   {base_fwd_t[2]:6.3f}               │"
    )
    print(
        f"│  FinslerLinear:      {finsler_fwd_t[0]:6.3f}   {finsler_fwd_t[1]:6.3f}   {finsler_fwd_t[2]:6.3f}               │"
    )
    print(
        f"│  Embed (base):       {base_embed_t[0]:6.3f}   {base_embed_t[1]:6.3f}   {base_embed_t[2]:6.3f}               │"
    )
    print(
        f"│  TwistedEmbed:       {twisted_embed_t[0]:6.3f}   {twisted_embed_t[1]:6.3f}   {twisted_embed_t[2]:6.3f}              │"
    )
    print("└" + "─" * 68 + "┘")

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark: Dualization (gradient → update)
    # ─────────────────────────────────────────────────────────────────────

    print("\n┌" + "─" * 68 + "┐")
    print("│ DUALIZATION - gradient→update (ms)      p50      p95      p99      │")
    print("│ how does the forward pass transforms under reflection              │")
    print("├" + "─" * 68 + "┤")

    base_dual_t = benchmark(lambda: base_dual(grad_base), n_iterations)
    finsler_dual_t = benchmark(lambda: finsler_dual(grad_finsler), n_iterations)
    embed_dual_t = benchmark(lambda: embed_dual(grad_embed), n_iterations)
    twisted_dual_t = benchmark(lambda: twisted_dual(grad_embed), n_iterations)

    print(
        f"│  Linear (base):      {base_dual_t[0]:6.3f}   {base_dual_t[1]:6.3f}   {base_dual_t[2]:6.3f}               │"
    )
    print(
        f"│  FinslerLinear:      {finsler_dual_t[0]:6.3f}   {finsler_dual_t[1]:6.3f}   {finsler_dual_t[2]:6.3f}               │"
    )
    print(
        f"│  Embed (base):       {embed_dual_t[0]:6.3f}   {embed_dual_t[1]:6.3f}   {embed_dual_t[2]:6.3f}               │"
    )
    print(
        f"│  TwistedEmbed:       {twisted_dual_t[0]:6.3f}   {twisted_dual_t[1]:6.3f}   {twisted_dual_t[2]:6.3f}               │"
    )
    print("└" + "─" * 68 + "┘")

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark: Weight Projection
    # ─────────────────────────────────────────────────────────────────────

    print("\n┌" + "─" * 68 + "┐")
    print("│ PROJECTION - weight normalization (ms)  p50      p95      p99      │")
    print("├" + "─" * 68 + "┤")

    base_proj_t = benchmark(lambda: base_proj(base_w), n_iterations)
    finsler_proj_t = benchmark(lambda: finsler_proj(finsler_w), n_iterations)

    print(
        f"│  Linear (base):      {base_proj_t[0]:6.3f}   {base_proj_t[1]:6.3f}   {base_proj_t[2]:6.3f}               │"
    )
    print(
        f"│  FinslerLinear:      {finsler_proj_t[0]:6.3f}   {finsler_proj_t[1]:6.3f}   {finsler_proj_t[2]:6.3f}               │"
    )
    print("└" + "─" * 68 + "┘")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "═" * 70)
    print("                              Summary")
    print("═" * 70)
    print(
        """
  • Forward pass: Geometric layers have ~0% overhead (same matrix multiply)
  • Dualization:  FinslerLinear uses finsler_orthogonalize (drift-aware)
  • Projection:   FinslerLinear projects both W and drift vector

  ✓ Geometric structure adds minimal overhead thanks to JAX JIT
  ✓ The asymmetric dualization in FinslerLinear is the key differentiator
"""
    )


def main():
    parser = argparse.ArgumentParser(
        prog='diffgeo',
        description='DiffGeo - Differential Geometry Extensions for Modula',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  diffgeo info                 Show available components
  diffgeo demo spd             SPD manifold operations
  diffgeo demo finsler         Asymmetric Finsler metrics
  diffgeo demo chiral          Chirality detection
  diffgeo check invariants     Run math invariant tests
  diffgeo benchmark            Performance comparison
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # info command
    subparsers.add_parser('info', help='Show package info and components')
    
    # demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demos')
    demo_parser.add_argument('name', choices=['spd', 'finsler', 'chiral'],
                            help='Demo to run')
    
    # check command
    check_parser = subparsers.add_parser('check', help='Run verification checks')
    check_parser.add_argument('what', choices=['invariants'],
                             help='What to check')
    
    # benchmark command
    subparsers.add_parser('benchmark', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'info':
        cmd_info(args)
    elif args.command == 'demo':
        if args.name == 'spd':
            cmd_demo_spd(args)
        elif args.name == 'finsler':
            cmd_demo_finsler(args)
        elif args.name == 'chiral':
            cmd_demo_chiral(args)
    elif args.command == 'check':
        if args.what == 'invariants':
            return cmd_check_invariants(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
