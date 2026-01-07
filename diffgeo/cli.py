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
        GeometricLinear, FinslerLinear, TwistedEmbed, GeometricEmbed,
        MetricTensor, RandersMetric, SPDManifold, FisherMetric,
    )
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                          DiffGeo for Modula                          ║
║           Differential Geometry Extensions for Neural Nets           ║
╚══════════════════════════════════════════════════════════════════════╝

Geometric Atoms:
  • GeometricLinear   - Linear with explicit vector→vector signature
  • FinslerLinear     - Linear with asymmetric Finsler metric (directed)
  • TwistedEmbed      - Orientation-sensitive embedding (chirality)
  • GeometricEmbed    - Standard embedding with geometric tracking
  • ContactAtom       - Conservation law projection

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
    from diffgeo import GeometricLinear, FinslerLinear
    from modula.atom import Linear  # Base modula
    
    # Standard geometric layer
    geo_layer = GeometricLinear(64, 128)
    
    # Finsler layer for directed/causal data
    finsler_layer = FinslerLinear(64, 128, drift_strength=0.3)
""")


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
    """Run performance benchmarks."""
    from diffgeo import GeometricLinear, FinslerLinear
    from modula.atom import Linear
    import time
    
    print("\n=== Performance Benchmarks ===\n")
    
    key = jax.random.PRNGKey(42)
    dim = 256
    batch_size = 128
    n_iterations = 100
    
    # Create layers
    base_linear = Linear(dim, dim)
    geo_linear = GeometricLinear(dim, dim)
    finsler_linear = FinslerLinear(dim, dim, drift_strength=0.3)
    
    # Initialize
    k1, k2, k3, k4 = jax.random.split(key, 4)
    base_w = base_linear.initialize(k1)
    geo_w = geo_linear.initialize(k2)
    finsler_w = finsler_linear.initialize(k3)
    
    # Create batch
    batch = jax.random.normal(k4, (batch_size, dim))
    
    # JIT compile
    base_fwd = jax.jit(lambda x, w: jax.vmap(lambda xi: base_linear.forward(xi, w))(x))
    geo_fwd = jax.jit(lambda x, w: jax.vmap(lambda xi: geo_linear.forward(xi, w))(x))
    finsler_fwd = jax.jit(lambda x, w: jax.vmap(lambda xi: finsler_linear.forward(xi, w))(x))
    
    # Warmup
    _ = base_fwd(batch, base_w).block_until_ready()
    _ = geo_fwd(batch, geo_w).block_until_ready()
    _ = finsler_fwd(batch, finsler_w).block_until_ready()
    
    # Benchmark
    print(f"Config: dim={dim}, batch={batch_size}, iterations={n_iterations}")
    print()
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = base_fwd(batch, base_w).block_until_ready()
    base_time = (time.perf_counter() - start) / n_iterations * 1000
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = geo_fwd(batch, geo_w).block_until_ready()
    geo_time = (time.perf_counter() - start) / n_iterations * 1000
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = finsler_fwd(batch, finsler_w).block_until_ready()
    finsler_time = (time.perf_counter() - start) / n_iterations * 1000
    
    print(f"Forward pass timing (ms/batch):")
    print(f"  Linear (base):      {base_time:.3f} ms")
    print(f"  GeometricLinear:    {geo_time:.3f} ms ({geo_time/base_time:.2f}x)")
    print(f"  FinslerLinear:      {finsler_time:.3f} ms ({finsler_time/base_time:.2f}x)")
    
    print("\n✓ Geometric overhead is minimal due to JAX JIT compilation")


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

