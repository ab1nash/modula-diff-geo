#!/usr/bin/env python3
"""
Run Fisher Geometry Benchmarks (FIXED Version)

Tests the REPAIRED Fisher Geometry Framework on real datasets:
- PhysioNet EEG (SPD covariance matrices)
- GHCN Climate (spatial data)

=============================================================================
KEY FIX: This version uses the CORRECTED Fisher models that compute
Fisher Information from DATA MANIFOLD (not neural network parameters).

PROBLEM WITH OLD FisherImputationModel:
  - Computed Fisher from NN gradients: E[∇_θ L ⊗ ∇_θ L]
  - This is Fisher on PARAMETER SPACE (wrong!)
  
NEW FIXED MODELS:
  - SPDFisherModel: Fisher from data covariance in tangent space (correct!)
  - SphericalFisherModel: Fisher with great-circle distance (correct!)
  
The key insight from research doc [2] Section 4:
  "For Gaussian N(μ, Σ), the Fisher metric on mean parameters is Σ^{-1}.
   This matches the Riemannian metric on SPD manifold."
=============================================================================

Compares approaches:
1. Modula (Euclidean) - baseline
2. SPD Tangent Space - hand-crafted manifold geometry
3. Fisher (OLD) - BROKEN Fisher (parameter space)
4. SPD Fisher (NEW) - FIXED Fisher (data manifold)
5. Spherical Fisher (NEW) - FIXED Fisher for S² data

Uses the shared multi-model benchmark infrastructure for consistent
JSON/figure output.

Usage:
    LD_LIBRARY_PATH=/opt/rocm/lib python run_fisher_benchmarks.py
    python run_fisher_benchmarks.py --quick  # Fast test
    python run_fisher_benchmarks.py --fixed-only  # Only run fixed models
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_fisher_benchmarks(mode: str = 'quick', datasets: list = None, 
                          fixed_only: bool = False):
    """
    Run benchmarks comparing Euclidean, SPD, and Fisher geometry.
    
    Uses the shared multi-model infrastructure for automatic
    JSON saving and figure generation.
    
    Args:
        mode: 'quick' for fast testing, 'standard' for full benchmark
        datasets: List of datasets to test
        fixed_only: If True, only run Fisher models (skip baselines)
    """
    import numpy as np
    
    from tests.realworld.benchmarks import (
        TrainingConfig,
        ModulaImputationModel,
        SPDTangentSpaceModel,
        SPDFisherModel,
        ExtractedFisherModel,
        train_model,
        train_spd_model,
        train_spd_fisher_model,
        train_extracted_fisher_model,
        evaluate_model,
        evaluate_spd_model,
        evaluate_spd_fisher_model,
        evaluate_extracted_fisher_model,
        # Multi-model infrastructure
        ModelConfig,
        run_multi_model_benchmark,
        save_multi_benchmark_results,
        print_multi_summary,
        ManifoldType,
    )
    from data.loaders import DatasetRegistry
    
    # Configure based on mode
    if mode == 'quick':
        config = TrainingConfig.quick()
        missing_fractions = [0.3, 0.6, 0.9]
        n_samples = 50
        n_runs = 1
    elif mode == 'full':
        config = TrainingConfig.full()
        missing_fractions = [0.15, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        n_samples = 150
        n_runs = 3
    else:
        config = TrainingConfig.standard()
        missing_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_samples = 150
        n_runs = 2
    
    datasets = datasets or ['physionet_eeg', 'ghcn_daily', 'cmu_mocap']
    
    # Define model configurations
    # Colors: Red=baseline, Green=hand-crafted, Purple=Fisher (data manifold)
    MODEL_COLORS = {
        'Modula': '#e74c3c',        # Red - Euclidean baseline
        'SPD Tangent': '#27ae60',   # Green - hand-crafted SPD geometry
        'SPD Fisher': '#9b59b6',    # Purple - Fisher on SPD data manifold
        'Extracted Fisher': '#3498db',  # Blue - universal learned geometry
    }
    
    print("=" * 70)
    print("FISHER GEOMETRY BENCHMARKS")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Datasets: {datasets}")
    print(f"Missing fractions: {missing_fractions}")
    print(f"Samples per dataset: {n_samples}")
    print(f"Runs per condition: {n_runs}")
    print("=" * 70)
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Load data
            loader = DatasetRegistry.get_loader(dataset_name)
            
            if dataset_name == 'physionet_eeg':
                dataset = loader.load_multiple_subjects()
                data = dataset.matrices[:n_samples]
                manifold_type = ManifoldType.SPD
                is_spd = True
                print(f"  Loaded {data.shape[0]} SPD matrices ({data.shape[1]}×{data.shape[2]})")
                
            elif dataset_name == 'ghcn_daily':
                dataset = loader.load_stations(n_stations=n_samples)
                data = np.column_stack([dataset.coordinates, dataset.values])
                manifold_type = ManifoldType.SPHERE
                is_spd = False
                print(f"  Loaded {len(data)} stations, {data.shape[1]} features")
                
            elif dataset_name == 'cmu_mocap':
                dataset = loader.load_motion("01", "01")
                # Flatten joint angles from (n_frames, n_joints, 3) to (n_frames, n_joints*3)
                joint_data = dataset.joint_angles[:n_samples * 4]
                data = joint_data.reshape(joint_data.shape[0], -1)
                manifold_type = ManifoldType.SO3
                is_spd = False
                print(f"  Loaded {len(data)} frames, {data.shape[1]} features")
            else:
                print(f"  Unknown dataset: {dataset_name}")
                continue
            
            # Define models to benchmark
            model_configs = []
            
            if not fixed_only:
                # Baseline: Euclidean (Modula)
                model_configs.append(
                    ModelConfig(
                        name='Modula',
                        model_class=ModulaImputationModel,
                        color='#e74c3c',
                    )
                )
            
                # Add SPD-specific model only for SPD data
                if is_spd:
                    model_configs.append(
                        ModelConfig(
                            name='SPD Tangent',
                            model_class=SPDTangentSpaceModel,
                            train_fn=train_spd_model,
                            eval_fn=evaluate_spd_model,
                            color='#27ae60',
                        )
                    )
                
                # Old FisherImputationModel removed - it was computing Fisher
                # in parameter space (wrong!) instead of data space
            
            # NEW FIXED Fisher models
            # 
            # KEY INSIGHT: Fisher Information is UNIVERSAL - it discovers geometry.
            # But the REPRESENTATION matters:
            #
            # For SPD data: We KNOW the structure, so use tangent space + Fisher
            #   - SPDFisherModel uses matrix log/exp (known structure)
            #   - Fisher metric computed in tangent space (learned from data)
            #
            # For unknown structure: ExtractedFisherModel discovers geometry
            #   - Uses DataGeometryExtractor to fit a distribution
            #   - Fisher metric emerges from covariance structure
            
            if is_spd:
                # SPD Fisher: Known structure (tangent space) + learned Fisher metric
                # Best of both: geometric guarantees + data-driven curvature
                model_configs.append(
                    ModelConfig(
                        name='SPD Fisher',
                        model_class=SPDFisherModel,
                        train_fn=train_spd_fisher_model,
                        eval_fn=evaluate_spd_fisher_model,
                        color='#9b59b6',
                    )
                )
            else:
                # Extracted Fisher: UNIVERSAL - learns geometry from data
                # For data where we DON'T know the manifold structure,
                # let Fisher Information discover it
                model_configs.append(
                    ModelConfig(
                        name='Extracted Fisher',
                        model_class=ExtractedFisherModel,
                        train_fn=train_extracted_fisher_model,
                        eval_fn=evaluate_extracted_fisher_model,
                        color='#3498db',
                    )
                )
            
            # Run benchmark using shared infrastructure
            results = run_multi_model_benchmark(
                data=data,
                dataset_name=dataset_name,
                model_configs=model_configs,
                missing_fractions=missing_fractions,
                config=config,
                manifold_type=manifold_type,
                n_runs=n_runs,
            )
            
            # Print summary
            print_multi_summary(results)
            
            # Save JSON and figures
            saved = save_multi_benchmark_results(results, MODEL_COLORS)
            print(f"\nSaved {len(saved)} files:")
            for key, path in saved.items():
                print(f"  {key}: {path}")
            
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {PROJECT_ROOT / 'results'}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisher Geometry Benchmarks")
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--full', action='store_true', help='Full benchmark mode')
    parser.add_argument('--datasets', nargs='+', 
                        default=['physionet_eeg', 'ghcn_daily', 'cmu_mocap'],
                        choices=['physionet_eeg', 'ghcn_daily', 'cmu_mocap'],
                        help='Datasets to test')
    parser.add_argument('--fisher-only', action='store_true',
                        help='Only run Fisher models (skip baseline comparisons)')
    
    args = parser.parse_args()
    
    if args.quick:
        mode = 'quick'
    elif args.full:
        mode = 'full'
    else:
        mode = 'standard'
    run_fisher_benchmarks(
        mode=mode, 
        datasets=args.datasets,
        fixed_only=args.fisher_only,
    )
