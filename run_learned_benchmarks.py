#!/usr/bin/env python3
"""
Run Learned Imputation Benchmarks

Trains and compares Modula (Euclidean) vs DiffGeo (Riemannian) 
imputation models with proper validation.

Usage:
    # For AMD ROCm GPU support, set LD_LIBRARY_PATH BEFORE Python:
    LD_LIBRARY_PATH=/opt/rocm/lib python run_learned_benchmarks.py
    
    # CPU mode (default if ROCm not configured):
    python run_learned_benchmarks.py                    # Standard (5 runs avg)
    python run_learned_benchmarks.py --quick            # Quick test (1 run)
    python run_learned_benchmarks.py --thorough         # Best results
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_hardware_info():
    """Print hardware information: GPU, CPU, and RAM."""
    import platform
    import subprocess
    
    print("=" * 70)
    print("HARDWARE INFORMATION")
    print("=" * 70)
    
    # CPU info
    cpu_info = platform.processor() or "Unknown"
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu_info = line.split(':')[1].strip()
                    break
    except:
        pass
    print(f"CPU: {cpu_info}")
    
    # RAM info
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    print(f"RAM: {mem_gb:.1f} GB")
                    break
    except:
        print("RAM: Unknown")
    
    # GPU info - check for AMD ROCm first, then NVIDIA
    gpu_info = None
    gpu_backend = None
    
    # Check JAX backend
    try:
        import jax
        devices = jax.devices()
        backend = jax.default_backend()
        print(f"JAX Backend: {backend}")
        print(f"JAX Devices: {[str(d) for d in devices]}")
        
        if backend == 'gpu':
            gpu_backend = 'GPU (JAX)'
        elif backend == 'cpu':
            gpu_backend = 'CPU only'
    except ImportError:
        print("JAX: Not installed")
    except Exception as e:
        print(f"JAX: Error detecting - {e}")
    
    # Check for AMD GPU via ROCm
    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Card series' in line or 'Card Series' in line:
                    # Parse the line to get just the GPU name
                    parts = line.split(':')
                    if len(parts) >= 2:
                        gpu_info = parts[-1].strip()
                    else:
                        gpu_info = line.strip()
                    break
            if not gpu_info:
                gpu_info = "AMD GPU (ROCm detected)"
            print(f"GPU: {gpu_info}")
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # Check for AMD GPU via lspci if rocm-smi not available
    if not gpu_info:
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or 'Display' in line or '3D' in line:
                        if 'AMD' in line or 'Radeon' in line:
                            gpu_info = line.split(':')[-1].strip()
                            print(f"GPU: {gpu_info}")
                            break
                        elif 'NVIDIA' in line:
                            gpu_info = line.split(':')[-1].strip()
                            print(f"GPU: {gpu_info}")
                            break
        except:
            pass
    
    if not gpu_info:
        print("GPU: Not detected (using CPU)")
    
    # Check if JAX is using the GPU
    try:
        import jax
        import jax.numpy as jnp
        # Try a small computation to verify GPU is being used
        x = jnp.ones((100, 100))
        _ = jnp.dot(x, x).block_until_ready()
        device = jax.devices()[0]
        using_gpu = device.platform != 'cpu'
        print(f"Compute Device: {device.platform.upper()} - {device}")
        
        # Warn if GPU available but not being used
        if gpu_info and not using_gpu:
            print("")
            print("⚠️  WARNING: GPU detected but JAX is using CPU!")
            print("   Ensure LD_LIBRARY_PATH includes /opt/rocm/lib")
            print("   ROCm plugin requires ~4-5GB RAM to load")
    except Exception as e:
        print(f"Compute Device: CPU (GPU not available: {e})")
    
    print("=" * 70)


def run_benchmarks(mode: str = 'standard',
                   n_epochs: int = None,
                   datasets: list = None,
                   n_runs: int = None):
    """
    Run learned imputation benchmarks.
    
    Args:
        mode: 'quick' (testing), 'standard' (default), 'thorough' (best results)
        n_epochs: Override epoch count (optional)
        datasets: Specific datasets to test
        n_runs: Number of runs to average over (default: 1 for quick, 5 otherwise)
    """
    import numpy as np
    
    from tests.realworld.benchmarks import (
        TrainingConfig,
        run_learned_benchmark,
        save_learned_benchmark_results,
        print_learned_summary,
    )
    from tests.realworld.benchmarks.manifold_integrity import ManifoldType
    from data.loaders import DatasetRegistry
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure training using presets
    if mode == 'quick':
        config = TrainingConfig.quick()
        missing_fractions = [0.3, 0.6, 0.9]
        default_runs = 1
    elif mode == 'thorough':
        config = TrainingConfig.thorough()
        missing_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        default_runs = 3
    else:  # standard
        config = TrainingConfig.standard()
        missing_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        default_runs = 2
    
    # Override epochs if specified
    if n_epochs is not None:
        config.n_epochs = n_epochs
    
    # Use specified n_runs or mode default
    n_runs = n_runs if n_runs is not None else default_runs
    
    # Print hardware info first
    print_hardware_info()
    
    print("\n" + "=" * 70)
    print(f"LEARNED IMPUTATION BENCHMARKS")
    print(f"Timestamp: {timestamp}")
    print(f"Mode: {mode}")
    print(f"Runs per condition: {n_runs} (for statistical averaging)")
    print(f"Max Epochs: {config.n_epochs}")
    print(f"Early Stop Patience: {config.early_stopping_patience}")
    print(f"LR Decay Patience: {config.lr_decay_patience}")
    print(f"Missing Fractions: {missing_fractions}")
    print("=" * 70)
    
    datasets = datasets or ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']
    
    # Sample sizes - larger for more robust benchmarks
    # Quick mode uses smaller samples for fast testing
    if mode == 'quick':
        n_eeg_samples = 50
        n_stations = 100
        n_mocap_frames = 200
    else:
        # Standard/thorough use larger samples for reliable results
        n_eeg_samples = 200
        n_stations = 300
        n_mocap_frames = 500
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Load data
            loader = DatasetRegistry.get_loader(dataset_name)
            
            manifold_type = None
            matrix_dim = None  # For MIS computation on SPD
            
            if dataset_name == 'physionet_eeg':
                dataset = loader.load_multiple_subjects()
                # Keep SPD matrices in original shape for MIS computation
                data = dataset.matrices[:n_eeg_samples]  # (n_samples, dim, dim)
                matrix_dim = data.shape[1]
                manifold_type = ManifoldType.SPD
                print(f"  Loaded {data.shape[0]} samples, matrix_dim={matrix_dim}")
                
            elif dataset_name == 'ghcn_daily':
                dataset = loader.load_stations(n_stations=n_stations)
                data = np.column_stack([dataset.coordinates, dataset.values])
                manifold_type = ManifoldType.SPHERE
                print(f"  Loaded {len(data)} stations, features={data.shape[1]}")
                
            elif dataset_name == 'cmu_mocap':
                motions = loader.load_multiple_motions(n_motions=5)
                # Concatenate and flatten
                data = np.concatenate([m.joint_angles.reshape(m.n_frames, -1) 
                                      for m in motions])[:n_mocap_frames]
                manifold_type = ManifoldType.SO3
                print(f"  Loaded {len(data)} frames, features={data.shape[1]}")
            
            # Run learned benchmark with multiple runs for averaging
            results = run_learned_benchmark(
                data=data,
                dataset_name=dataset_name,
                missing_fractions=missing_fractions,
                config=config,
                manifold_type=manifold_type,
                n_runs=n_runs,
            )
            
            # Print summary
            print_learned_summary(results)
            
            # Save results
            saved = save_learned_benchmark_results(results)
            print(f"\nSaved {len(saved)} files")
            
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    print(f"\nResults saved to: {PROJECT_ROOT / 'results'}")
    print(f"Figures saved to: {PROJECT_ROOT / 'results' / 'figures'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run Learned Imputation Benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  quick     - 200 epochs, 30 patience, 1 run (for testing)
  standard  - 1000 epochs, 75 patience, 5 runs (balanced)
  thorough  - 3000 epochs, 150 patience, 5 runs (best results)

Sample Sizes:
  quick     - 50 EEG, 100 stations, 200 mocap frames
  standard  - 200 EEG, 300 stations, 500 mocap frames
  thorough  - 200 EEG, 300 stations, 500 mocap frames

Examples:
  python run_learned_benchmarks.py                    # Standard (5 runs avg)
  python run_learned_benchmarks.py --quick            # Quick test (1 run)
  python run_learned_benchmarks.py --thorough         # Best results
  python run_learned_benchmarks.py --runs 10          # More runs for lower variance
  python run_learned_benchmarks.py --epochs 2000      # Custom epochs
  python run_learned_benchmarks.py --datasets physionet_eeg  # Single dataset
"""
    )
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max training epochs')
    parser.add_argument('--runs', type=int, default=None,
                       help='Number of runs to average over (default: 1 for quick, 5 otherwise)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (200 epochs, 1 run)')
    parser.add_argument('--thorough', action='store_true',
                       help='Thorough mode (3000 epochs, 5 runs)')
    parser.add_argument('--datasets', nargs='+',
                       choices=['physionet_eeg', 'ghcn_daily', 'cmu_mocap'],
                       help='Specific datasets to benchmark')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        mode = 'quick'
    elif args.thorough:
        mode = 'thorough'
    else:
        mode = 'standard'
    
    run_benchmarks(
        mode=mode,
        n_epochs=args.epochs,
        datasets=args.datasets,
        n_runs=args.runs,
    )


if __name__ == '__main__':
    main()

