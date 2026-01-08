#!/usr/bin/env python3
"""
Main Benchmark Runner for Geometric Imputation

Compares modula (baseline Euclidean) vs diffgeo (geometric) imputation
on real-world datasets spanning different manifold types.

Usage:
    python run_benchmarks.py --download          # Download available datasets
    python run_benchmarks.py --benchmark         # Run benchmarks
    python run_benchmarks.py --download --benchmark  # Both
    python run_benchmarks.py --quick             # Quick test run
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_datasets(force: bool = False):
    """Download all available datasets."""
    print("\n" + "=" * 70)
    print("DOWNLOADING REAL-WORLD GEOMETRIC DATASETS")
    print("=" * 70)
    
    from data.download import download_all_available, check_dataset_status
    
    # Show current status
    print("\nCurrent dataset status:")
    status = check_dataset_status()
    for name, info in status.items():
        marker = "✓" if info['exists'] else ("⬇" if info['downloadable'] else "✗")
        print(f"  {marker} {name}: {info.get('n_files', 0)} files, {info.get('size_mb', 0):.1f} MB")
    
    # Download
    print("\nDownloading...")
    results = download_all_available(force=force)
    
    return results


def run_benchmarks(datasets: list = None, quick: bool = False):
    """Run imputation benchmarks on specified datasets."""
    print("\n" + "=" * 70)
    print("RUNNING GEOMETRIC IMPUTATION BENCHMARKS")
    print("=" * 70)
    
    from tests.realworld.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
    from tests.realworld.benchmarks.visualization import save_benchmark_figures
    from data.loaders import DatasetRegistry
    
    if datasets is None:
        datasets = ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Configure benchmark
            if quick:
                config = BenchmarkConfig(
                    dataset_name=dataset_name,
                    missing_fractions=[0.1, 0.3, 0.5, 0.7, 0.9],
                    n_trials=2,
                )
            else:
                config = BenchmarkConfig(
                    dataset_name=dataset_name,
                    missing_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                    n_trials=5,
                )
            
            runner = BenchmarkRunner(config)
            
            # Load data
            print(f"\nLoading {dataset_name} data...")
            loader = DatasetRegistry.get_loader(dataset_name)
            
            if dataset_name == 'physionet_eeg':
                dataset = loader.load_multiple_subjects()
                data = dataset.matrices[:100]  # Limit for speed
                print(f"  Loaded {len(data)} SPD matrices of size {data.shape[1]}x{data.shape[2]}")
                
            elif dataset_name == 'ghcn_daily':
                dataset = loader.load_stations(n_stations=50)
                # Combine coordinates and values
                data = np.column_stack([dataset.coordinates, dataset.values])
                print(f"  Loaded {len(data)} stations with {data.shape[1]} features")
                
            elif dataset_name == 'cmu_mocap':
                motions = loader.load_multiple_motions(n_motions=3)
                # Use first motion
                data = motions[0].joint_angles[:200]  # Limit frames
                print(f"  Loaded {len(data)} frames with {data.shape[1]} joints")
            
            # Run benchmark
            results = runner.run(data)
            
            # Print summary
            results.print_summary()
            
            # Save results
            results.save()
            
            # Generate figures
            print("\nGenerating figures...")
            saved_figs = save_benchmark_figures(results)
            print(f"  Saved {len(saved_figs)} figures")
            
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"ERROR benchmarking {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def print_manual_download_info():
    """Print information about datasets requiring manual download."""
    print("\n" + "=" * 70)
    print("DATASETS REQUIRING MANUAL DOWNLOAD")
    print("=" * 70)
    
    manual_datasets = [
        {
            'name': 'TUM VI Benchmark',
            'manifold': 'Lie Group SE(3)',
            'url': 'https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset',
            'reason': 'Large video/IMU data (~15GB per sequence)',
            'format': 'Images + IMU raw data',
        },
        {
            'name': 'UZH-FPV Drone Racing',
            'manifold': 'Lie Group SE(3)',
            'url': 'https://fpv.ifi.uzh.ch/',
            'reason': 'Large video/IMU/event data',
            'format': 'Images + IMU + Event camera',
        },
        {
            'name': 'Google Smartphone Decimeter',
            'manifold': 'SE(3) / Signal Space',
            'url': 'https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge/data',
            'reason': 'Requires Kaggle API authentication',
            'format': 'Raw GNSS logs',
        },
        {
            'name': 'Stanford HARDI',
            'manifold': 'SPD (diffusion tensor)',
            'url': 'https://purl.stanford.edu/ng782rw8378',
            'reason': 'Large MRI data (~2GB)',
            'format': 'NIfTI volumes',
        },
        {
            'name': 'Redwood 3D Scans',
            'manifold': '2-Manifold (Mesh)',
            'url': 'http://redwood-data.org/3dscan/',
            'reason': 'Large 3D mesh data',
            'format': 'PLY / RGB-D frames',
        },
    ]
    
    for ds in manual_datasets:
        print(f"\n{ds['name']}")
        print(f"  Manifold: {ds['manifold']}")
        print(f"  URL: {ds['url']}")
        print(f"  Reason: {ds['reason']}")
        print(f"  Format: {ds['format']}")
    
    print("\n" + "-" * 70)
    print("To use these datasets after manual download:")
    print("  1. Download from the URL above")
    print("  2. Extract to: modula-diff-geo/data/<dataset_name>/")
    print("  3. Run benchmarks with: python run_benchmarks.py --benchmark")


def main():
    parser = argparse.ArgumentParser(
        description='Geometric Imputation Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --download          Download all available datasets
  python run_benchmarks.py --benchmark         Run full benchmarks
  python run_benchmarks.py --quick             Quick test run
  python run_benchmarks.py --download --quick  Download and quick test
  python run_benchmarks.py --info              Show manual download info
"""
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Download available datasets')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarks')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced settings')
    parser.add_argument('--info', action='store_true',
                       help='Show info about manual downloads')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download existing files')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['physionet_eeg', 'ghcn_daily', 'cmu_mocap'],
                       help='Specific datasets to benchmark')
    
    args = parser.parse_args()
    
    # Default: show help if no args
    if not any([args.download, args.benchmark, args.quick, args.info]):
        parser.print_help()
        print("\nRunning quick benchmark as default...")
        args.quick = True
        args.benchmark = True
    
    # Import numpy here (after argparse)
    global np
    import numpy as np
    
    if args.info:
        print_manual_download_info()
    
    if args.download:
        download_datasets(force=args.force)
    
    if args.benchmark or args.quick:
        run_benchmarks(datasets=args.datasets, quick=args.quick)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    # Show what to do next
    print("\nResults saved to: modula-diff-geo/results/")
    print("Figures saved to: modula-diff-geo/results/figures/")
    
    if not args.info:
        print("\nFor manual download datasets, run:")
        print("  python run_benchmarks.py --info")


if __name__ == '__main__':
    main()

