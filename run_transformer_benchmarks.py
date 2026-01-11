#!/usr/bin/env python3
"""
Geometric Transformer Benchmarks

Compares GeometricGPT vs standard Modula GPT on tasks where geometry matters:
1. Music Direction - Detect forward vs reversed sequences
2. Reaction Direction - Predict chemical reaction direction  
3. Protein Structure - Secondary structure prediction

Uses same infrastructure as run_benchmarks.py for consistency.

Usage:
    python run_transformer_benchmarks.py --quick          # Quick test (3 epochs)
    python run_transformer_benchmarks.py                  # Standard (15 epochs)
    python run_transformer_benchmarks.py --benchmark music  # Single benchmark
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Reuse existing training config pattern
from tests.realworld.benchmarks import TrainingConfig

# Import models - use StandardGPTJIT for fair JIT comparison
from diffgeo import GeometricGPT, StandardGPTJIT


# =============================================================================
# Results Data Classes
# =============================================================================

@dataclass
class ModelResult:
    """Result from training a single model."""
    name: str
    test_acc: float
    train_time: float
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    extra_metrics: Dict = field(default_factory=dict)


@dataclass 
class BenchmarkResult:
    """Result from a benchmark comparing multiple models."""
    benchmark_name: str
    timestamp: str
    model_results: List[ModelResult]
    config: Dict
    
    def best_model(self) -> ModelResult:
        return max(self.model_results, key=lambda r: r.test_acc)
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.benchmark_name}")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Acc':>10} {'Time':>10}")
        print("-" * 60)
        for r in self.model_results:
            print(f"{r.name:<30} {r.test_acc:>10.4f} {r.train_time:>9.1f}s")
        print("-" * 60)
        best = self.best_model()
        print(f"Best: {best.name} ({best.test_acc:.4f})")
    
    def save(self, results_dir: Path = None):
        results_dir = results_dir or PROJECT_ROOT / "results" / "json"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        path = results_dir / f"{self.timestamp}_{self.benchmark_name}.json"
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return path


# =============================================================================
# Data Generation (Cached)
# =============================================================================

CACHE_DIR = PROJECT_ROOT / "benchmarks" / ".cache"


def get_cached_data(name: str, generate_fn, key, config: dict):
    """Load data from cache or generate."""
    import hashlib
    import pickle
    
    CACHE_DIR.mkdir(exist_ok=True)
    config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"{name}_{config_hash}.pkl"
    
    if cache_path.exists():
        print(f"  Loading from cache...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return {k: jnp.array(v) for k, v in data['train'].items()}, \
               {k: jnp.array(v) for k, v in data['test'].items()}
    
    print(f"  Generating data...")
    train, test = generate_fn(key)
    
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train': {k: np.array(v) for k, v in train.items()},
            'test': {k: np.array(v) for k, v in test.items()},
        }, f)
    
    return train, test


# =============================================================================
# Data Generators
# =============================================================================

def generate_music_data(key, num_train=2000, num_test=500, seq_len=32):
    """Generate music direction detection data."""
    k1, k2, k3 = jax.random.split(key, 3)
    
    def make_split(k, n):
        k1, k2, k3 = jax.random.split(k, 3)
        # Generate melodies with temporal structure
        base = jax.random.randint(k1, (n, 1), 48, 72)
        intervals = jax.random.randint(k2, (n, seq_len - 1), -3, 4)
        contour = jnp.cumsum(intervals, axis=1)
        melody = jnp.clip(jnp.concatenate([base, base + contour], axis=1), 0, 127)
        
        # Reverse half
        mask = jax.random.bernoulli(k3, 0.5, (n,))
        sequences = jnp.where(mask[:, None], jnp.flip(melody, axis=1), melody)
        return {"sequences": sequences.astype(jnp.int32), "labels": mask.astype(jnp.int32)}
    
    return make_split(k2, num_train), make_split(k3, num_test)


def generate_reaction_data(key, num_train=2000, num_test=500, seq_len=24, vocab=64):
    """Generate reaction direction data."""
    k1, k2 = jax.random.split(key)
    
    def make_split(k, n):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        reactants = jax.random.randint(k1, (n, 10), 1, vocab)
        is_forward = jax.random.bernoulli(k2, 0.5, (n,))
        shift = jax.random.randint(k3, (n, 1), 5, 15)
        noise = jax.random.randint(k4, (n, 10), -5, 6)
        
        products = jnp.where(is_forward[:, None],
                            jnp.clip(reactants - shift + noise, 1, vocab - 1),
                            jnp.clip(reactants + shift + noise // 2, 1, vocab - 1))
        
        sep = jnp.zeros((n, 1), dtype=jnp.int32)
        pad = jnp.zeros((n, 3), dtype=jnp.int32)
        sequences = jnp.concatenate([reactants, sep, products, pad], axis=1)
        
        return {"sequences": sequences, "labels": is_forward.astype(jnp.int32)}
    
    return make_split(k1, num_train), make_split(k2, num_test)


def generate_protein_data(key, num_train=2000, num_test=500, seq_len=32):
    """Generate protein structure data."""
    k1, k2 = jax.random.split(key)
    
    HELIX_PROP = jnp.array([0.83, 0.79, 0.73, 0.80, 0.77, 0.96, 1.00, 0.53, 0.80, 0.80,
                           0.87, 0.93, 0.93, 0.87, 0.53, 0.73, 0.77, 0.80, 0.67, 0.80])
    SHEET_PROP = jnp.array([0.72, 0.79, 0.65, 0.62, 0.97, 0.82, 0.62, 0.72, 0.82, 1.00,
                           0.87, 0.69, 0.79, 0.90, 0.54, 0.72, 0.97, 0.90, 1.00, 1.00])
    HYDRO = jnp.array([0.70, 0.00, 0.11, 0.11, 0.78, 0.11, 0.11, 0.46, 0.14, 1.00,
                      0.92, 0.07, 0.71, 0.81, 0.32, 0.41, 0.36, 0.40, 0.36, 0.97])
    
    def make_split(k, n):
        k1, k2, k3 = jax.random.split(k, 3)
        sequences = jax.random.randint(k1, (n, seq_len), 0, 20)
        
        helix = HELIX_PROP[sequences]
        sheet = SHEET_PROP[sequences]
        hydro = HYDRO[sequences]
        
        # Simple convolution for windowed averages
        kernel = jnp.ones(5) / 5
        helix_avg = jax.vmap(lambda x: jnp.convolve(x, kernel, mode='same'))(helix)
        sheet_avg = jax.vmap(lambda x: jnp.convolve(x, kernel, mode='same'))(sheet)
        hydro_avg = jax.vmap(lambda x: jnp.convolve(x, kernel, mode='same'))(hydro)
        
        is_helix = helix_avg > 0.82
        is_sheet = (hydro_avg > 0.7) & (sheet_avg > 0.85) & ~is_helix
        structures = jnp.where(is_helix, 0, jnp.where(is_sheet, 1, 2))
        
        # Add noise
        noise_mask = jax.random.bernoulli(k2, 0.1, (n, seq_len))
        noise_struct = jax.random.randint(k3, (n, seq_len), 0, 3)
        structures = jnp.where(noise_mask, noise_struct, structures)
        
        return {"sequences": sequences.astype(jnp.int32), 
                "structures": structures.astype(jnp.int32)}
    
    return make_split(k1, num_train), make_split(k2, num_test)


# =============================================================================
# Model Training
# =============================================================================

def create_models(benchmark: str, vocab_size: int):
    """Create models to compare for a benchmark."""
    d_embed, num_heads, num_blocks = 64, 4, 2
    d_query = d_value = d_embed // num_heads
    
    # Use StandardGPTJIT for fair JIT comparison (same RopeJIT infrastructure)
    models = {
        "Standard GPT": StandardGPTJIT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks),
    }
    
    if benchmark == "music":
        models["GeometricGPT (+1)"] = GeometricGPT(vocab_size, num_heads, d_embed, 
                                                    d_query, d_value, num_blocks, orientation=+1.0)
        models["GeometricGPT (-1)"] = GeometricGPT(vocab_size, num_heads, d_embed,
                                                    d_query, d_value, num_blocks, orientation=-1.0)
    elif benchmark == "reaction":
        for drift in [0.0, 0.3, 0.5]:
            models[f"GeometricGPT (drift={drift})"] = GeometricGPT(
                vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, drift_strength=drift)
    elif benchmark == "protein":
        models["GeometricGPT (L-form)"] = GeometricGPT(vocab_size, num_heads, d_embed,
                                                        d_query, d_value, num_blocks, orientation=+1.0)
        models["GeometricGPT (D-form)"] = GeometricGPT(vocab_size, num_heads, d_embed,
                                                        d_query, d_value, num_blocks, orientation=-1.0)
    
    return models


def train_model(model, train_data, test_data, n_epochs, batch_size, benchmark_type, model_name=""):
    """Train a single model and return results."""
    key = jax.random.PRNGKey(42)
    weights = model.initialize(key)
    
    is_classification = benchmark_type in ["music", "reaction"]
    is_music = benchmark_type == "music"
    is_protein = benchmark_type == "protein"
    
    def compute_loss(weights, sequences, labels_or_structures):
        logits = model.forward(sequences, weights)
        
        if is_classification:
            if is_music:
                pooled = jnp.mean(logits, axis=1)[:, :2]
            else:  # reaction
                pooled = logits[:, -1, :2]
            log_probs = jax.nn.log_softmax(pooled)
            loss = -jnp.mean(log_probs[jnp.arange(len(labels_or_structures)), labels_or_structures])
            acc = jnp.mean(jnp.argmax(pooled, axis=1) == labels_or_structures)
        else:  # protein (sequence labeling)
            struct_logits = logits[:, :, :3]
            log_probs = jax.nn.log_softmax(struct_logits, axis=-1)
            batch_idx = jnp.arange(sequences.shape[0])[:, None]
            seq_idx = jnp.arange(sequences.shape[1])[None, :]
            loss = -jnp.mean(log_probs[batch_idx, seq_idx, labels_or_structures])
            acc = jnp.mean(jnp.argmax(struct_logits, axis=-1) == labels_or_structures)
        
        return loss, acc
    
    # JIT compile the training step for speed
    @jax.jit
    def train_step(weights, batch_seqs, batch_labels, lr):
        (loss, acc), grads = jax.value_and_grad(
            lambda w: compute_loss(w, batch_seqs, batch_labels), has_aux=True
        )(weights)
        dualized = model.dualize(grads)
        new_weights = [w - lr * d for w, d in zip(weights, dualized)]
        return new_weights, loss, acc
    
    train_losses, train_accs = [], []
    num_samples = train_data["sequences"].shape[0]
    label_key = "labels" if is_classification else "structures"
    n_batches = num_samples // batch_size
    
    # Learning rate with cosine decay
    base_lr = 0.01
    
    start_time = time.time()
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(n_epochs), desc=f"  {model_name[:20]:<20}", 
                      unit="epoch", leave=True, ncols=80)
    
    for epoch in epoch_pbar:
        epoch_loss, epoch_acc = 0.0, 0.0
        
        # Cosine learning rate decay
        lr = base_lr * 0.5 * (1 + jnp.cos(jnp.pi * epoch / n_epochs))
        
        # Shuffle data each epoch
        shuffle_key = jax.random.fold_in(key, epoch)
        perm = jax.random.permutation(shuffle_key, num_samples)
        shuffled_seqs = train_data["sequences"][perm]
        shuffled_labels = train_data[label_key][perm]
        
        for i in range(n_batches):
            batch_seqs = shuffled_seqs[i*batch_size:(i+1)*batch_size]
            batch_labels = shuffled_labels[i*batch_size:(i+1)*batch_size]
            
            weights, loss, acc = train_step(weights, batch_seqs, batch_labels, lr)
            
            epoch_loss += float(loss)
            epoch_acc += float(acc)
        
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        
        # Update progress bar with current metrics
        epoch_pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc:.3f}", lr=f"{float(lr):.4f}")
    
    # Evaluate on FULL test set (not just first batch!)
    test_seqs = test_data["sequences"]
    test_labels = test_data[label_key]
    _, test_acc = compute_loss(weights, test_seqs, test_labels)
    
    return ModelResult(
        name="",  # Set by caller
        test_acc=float(test_acc),
        train_time=time.time() - start_time,
        train_losses=train_losses,
        train_accs=train_accs,
    )


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_music_benchmark(n_epochs=15, batch_size=64) -> BenchmarkResult:
    """Run music direction detection benchmark."""
    print("\n" + "="*60)
    print("BENCHMARK: Music Direction Detection")
    print("="*60)
    
    config = {"num_train": 2000, "num_test": 500, "seq_len": 32}
    train_data, test_data = get_cached_data(
        "music", lambda k: generate_music_data(k, **config),
        jax.random.PRNGKey(0), config
    )
    print(f"  Train: {train_data['sequences'].shape[0]}, Test: {test_data['sequences'].shape[0]}")
    
    models = create_models("music", vocab_size=128)
    results = []
    
    for name, model in models.items():
        print(f"\n{name}:")
        result = train_model(model, train_data, test_data, n_epochs, batch_size, "music", name)
        result.name = name
        results.append(result)
        print(f"  → Test Accuracy: {result.test_acc:.4f} ({result.train_time:.1f}s)")
    
    return BenchmarkResult(
        benchmark_name="music_direction",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        model_results=results,
        config={"n_epochs": n_epochs, "batch_size": batch_size, **config},
    )


def run_reaction_benchmark(n_epochs=15, batch_size=64) -> BenchmarkResult:
    """Run chemical reaction direction benchmark."""
    print("\n" + "="*60)
    print("BENCHMARK: Chemical Reaction Direction")
    print("="*60)
    
    config = {"num_train": 2000, "num_test": 500, "seq_len": 24, "vocab": 64}
    train_data, test_data = get_cached_data(
        "reaction", lambda k: generate_reaction_data(k, **config),
        jax.random.PRNGKey(0), config
    )
    print(f"  Train: {train_data['sequences'].shape[0]}, Test: {test_data['sequences'].shape[0]}")
    
    models = create_models("reaction", vocab_size=64)
    results = []
    
    for name, model in models.items():
        print(f"\n{name}:")
        result = train_model(model, train_data, test_data, n_epochs, batch_size, "reaction", name)
        result.name = name
        results.append(result)
        print(f"  → Test Accuracy: {result.test_acc:.4f} ({result.train_time:.1f}s)")
    
    return BenchmarkResult(
        benchmark_name="reaction_direction",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        model_results=results,
        config={"n_epochs": n_epochs, "batch_size": batch_size, **config},
    )


def run_protein_benchmark(n_epochs=15, batch_size=64) -> BenchmarkResult:
    """Run protein structure prediction benchmark."""
    print("\n" + "="*60)
    print("BENCHMARK: Protein Secondary Structure")
    print("="*60)
    
    config = {"num_train": 2000, "num_test": 500, "seq_len": 32}
    train_data, test_data = get_cached_data(
        "protein", lambda k: generate_protein_data(k, **config),
        jax.random.PRNGKey(0), config
    )
    print(f"  Train: {train_data['sequences'].shape[0]}, Test: {test_data['sequences'].shape[0]}")
    
    models = create_models("protein", vocab_size=21)
    results = []
    
    for name, model in models.items():
        print(f"\n{name}:")
        result = train_model(model, train_data, test_data, n_epochs, batch_size, "protein", name)
        result.name = name
        results.append(result)
        print(f"  → Test Accuracy: {result.test_acc:.4f} ({result.train_time:.1f}s)")
    
    return BenchmarkResult(
        benchmark_name="protein_structure",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        model_results=results,
        config={"n_epochs": n_epochs, "batch_size": batch_size, **config},
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Geometric Transformer Benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick test (3 epochs)')
    parser.add_argument('--benchmark', choices=['music', 'reaction', 'protein', 'all'],
                       default='all', help='Which benchmark to run')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    args = parser.parse_args()
    
    n_epochs = args.epochs or (3 if args.quick else 15)
    
    benchmarks = {
        'music': run_music_benchmark,
        'reaction': run_reaction_benchmark,
        'protein': run_protein_benchmark,
    }
    
    to_run = list(benchmarks.keys()) if args.benchmark == 'all' else [args.benchmark]
    all_results = []
    
    print(f"\nRunning {len(to_run)} benchmark(s) with {n_epochs} epochs each\n")
    
    for idx, name in enumerate(to_run, 1):
        print(f"[{idx}/{len(to_run)}] ", end="")
        result = benchmarks[name](n_epochs=n_epochs)
        result.print_summary()
        saved = result.save()
        print(f"Saved to: {saved}")
        all_results.append(result)
    
    print("\n" + "="*60)
    print("ALL BENCHMARKS COMPLETE")
    print("="*60)
    
    for r in all_results:
        best = r.best_model()
        baseline = next((m for m in r.model_results if "Standard" in m.name), None)
        if baseline and best.name != baseline.name:
            improvement = (best.test_acc - baseline.test_acc) / baseline.test_acc * 100
            print(f"{r.benchmark_name}: {best.name} beats baseline by {improvement:+.1f}%")
        else:
            print(f"{r.benchmark_name}: Best = {best.name} ({best.test_acc:.4f})")


if __name__ == '__main__':
    main()

