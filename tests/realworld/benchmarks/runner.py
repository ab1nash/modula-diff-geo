"""
Benchmark Runner for Geometric Imputation Experiments

Orchestrates data loading, masking, imputation, and evaluation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import json
import time

from .imputation import (
    ImputationBenchmark,
    ImputationMethod,
    ImputationResult,
    SPDImputationMethods,
    SphericalImputationMethods,
    MocapImputationMethods,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.realworld.utils.masking import DataMasker, MaskPattern


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    dataset_name: str
    missing_fractions: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    mask_patterns: List[MaskPattern] = field(default_factory=lambda: [MaskPattern.UNIFORM_RANDOM])
    n_trials: int = 5
    seed: int = 42
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent.parent / "results"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class MethodResult:
    """Results for a single method across trials."""
    method: ImputationMethod
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    manifold_error_mean: float
    manifold_error_std: float
    mis_mean: float  # Manifold Integrity Score (new!)
    mis_std: float
    validity_rate: float
    runtime_mean: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config: BenchmarkConfig
    results_by_fraction: Dict[float, Dict[str, MethodResult]]
    baseline_methods: List[str]
    geometric_methods: List[str]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'dataset_name': self.config.dataset_name,
                'missing_fractions': self.config.missing_fractions,
                'mask_patterns': [p.value for p in self.config.mask_patterns],
                'n_trials': self.config.n_trials,
                'seed': self.config.seed,
            },
            'results': {
                str(frac): {
                    method: {
                        'rmse_mean': r.rmse_mean,
                        'rmse_std': r.rmse_std,
                        'mae_mean': r.mae_mean,
                        'mae_std': r.mae_std,
                        'manifold_error_mean': r.manifold_error_mean,
                        'manifold_error_std': r.manifold_error_std,
                        'mis_mean': r.mis_mean,
                        'mis_std': r.mis_std,
                        'validity_rate': r.validity_rate,
                        'runtime_mean': r.runtime_mean,
                    }
                    for method, r in methods.items()
                }
                for frac, methods in self.results_by_fraction.items()
            },
            'baseline_methods': self.baseline_methods,
            'geometric_methods': self.geometric_methods,
            'timestamp': self.timestamp,
        }
    
    def save(self, filepath: Optional[Path] = None):
        """Save results to JSON with timestamp prefix."""
        if filepath is None:
            # Timestamp prefix for easy sorting
            filepath = self.config.output_dir / f"{self.timestamp}_{self.config.dataset_name}_closedform.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary table of results with neutral formatting."""
        print("\n" + "=" * 80)
        print(f"Benchmark Results: {self.config.dataset_name}")
        print("=" * 80)
        
        # Identify best method per fraction
        for frac in sorted(self.results_by_fraction.keys()):
            methods = self.results_by_fraction[frac]
            
            print(f"\nMissing Fraction: {frac:.0%}")
            print("-" * 90)
            print(f"{'#':<3} {'Method':<25} {'Type':<12} {'RMSE':>10} {'MAE':>10} {'MIS':>10} {'Valid':>8}")
            print("-" * 90)
            
            # Sort by RMSE - best first
            sorted_methods = sorted(methods.items(), key=lambda x: x[1].rmse_mean)
            
            for rank, (method_name, result) in enumerate(sorted_methods, 1):
                # Determine method type using neutral labels
                if method_name in self.geometric_methods:
                    method_type = "Geometric"
                elif method_name in self.baseline_methods:
                    method_type = "Euclidean"
                else:
                    method_type = "Other"
                
                print(f"{rank:<3} {method_name:<25} {method_type:<12} "
                      f"{result.rmse_mean:>10.4f} "
                      f"{result.mae_mean:>10.4f} "
                      f"{result.mis_mean:>10.4f} "
                      f"{result.validity_rate:>7.0%}")
        
        print("\n" + "-" * 90)
        print("Method Types: Geometric = uses manifold structure, Euclidean = flat space methods")
        print("MIS = Manifold Integrity Score (lower is better, 0 = perfect)")
        
        # Report wins by category (neutral)
        self._print_category_summary()
    
    def _print_category_summary(self):
        """Print a neutral summary of method performance by category."""
        geo_wins = 0
        euc_wins = 0
        total = 0
        
        for frac, methods in self.results_by_fraction.items():
            sorted_methods = sorted(methods.items(), key=lambda x: x[1].rmse_mean)
            if sorted_methods:
                best_method = sorted_methods[0][0]
                total += 1
                if best_method in self.geometric_methods:
                    geo_wins += 1
                elif best_method in self.baseline_methods:
                    euc_wins += 1
        
        print(f"\nSummary: Best method by missing fraction")
        print(f"  Geometric methods won: {geo_wins}/{total}")
        print(f"  Euclidean methods won: {euc_wins}/{total}")
        print(f"  Other methods won: {total - geo_wins - euc_wins}/{total}")


class BenchmarkRunner:
    """
    Run imputation benchmarks comparing modula vs diffgeo approaches.
    """
    
    MANIFOLD_BENCHMARKS = {
        'physionet_eeg': SPDImputationMethods,
        'ghcn_daily': SphericalImputationMethods,
        'cmu_mocap': MocapImputationMethods,
    }
    
    # Euclidean methods - both simple and competitive baselines
    BASELINE_METHODS = {
        # Simple baselines
        ImputationMethod.ZERO_FILL,
        ImputationMethod.MEAN_FILL,
        ImputationMethod.LINEAR_INTERP,
        ImputationMethod.NEAREST_NEIGHBOR,
        # Competitive Euclidean baselines
        ImputationMethod.SVD_IMPUTE,
        ImputationMethod.ITERATIVE_SVD,
        ImputationMethod.KNN_WEIGHTED,
        ImputationMethod.SPLINE_INTERP,
    }
    
    # Geometric methods - manifold-aware
    GEOMETRIC_METHODS = {
        ImputationMethod.LOG_EUCLIDEAN,
        ImputationMethod.FRECHET_MEAN,
        ImputationMethod.GEODESIC_INTERP,
        ImputationMethod.PARALLEL_TRANSPORT,
    }
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmark = self._get_benchmark()
        self.rng = np.random.default_rng(config.seed)
    
    def _get_benchmark(self) -> ImputationBenchmark:
        """Get appropriate benchmark class for dataset."""
        if self.config.dataset_name not in self.MANIFOLD_BENCHMARKS:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}. "
                           f"Available: {list(self.MANIFOLD_BENCHMARKS.keys())}")
        
        return self.MANIFOLD_BENCHMARKS[self.config.dataset_name]()
    
    def run(self, data: np.ndarray) -> BenchmarkResults:
        """
        Run complete benchmark on provided data.
        
        Args:
            data: Data array appropriate for the manifold type
                  - SPD: (n_samples, dim, dim)
                  - Spherical: (n_points, 2 + n_features)
                  - Mocap: (n_frames, n_joints, 3)
                  
        Returns:
            BenchmarkResults with all method comparisons
        """
        print(f"\nRunning benchmark: {self.config.dataset_name}")
        print(f"Data shape: {data.shape}")
        print(f"Missing fractions: {self.config.missing_fractions}")
        print(f"Trials per setting: {self.config.n_trials}")
        print("-" * 50)
        
        available_methods = self.benchmark.get_available_methods()
        
        results_by_fraction = {}
        
        for frac in self.config.missing_fractions:
            print(f"\nMissing fraction: {frac:.0%}")
            
            method_results = {}
            
            for method in available_methods:
                trial_results = self._run_trials(data, frac, method)
                
                if trial_results:
                    method_results[method.value] = self._aggregate_trials(
                        method, trial_results
                    )
                    print(f"  {method.value}: RMSE={method_results[method.value].rmse_mean:.4f}")
            
            results_by_fraction[frac] = method_results
        
        # Categorize methods
        baseline_methods = [m.value for m in available_methods if m in self.BASELINE_METHODS]
        geometric_methods = [m.value for m in available_methods if m in self.GEOMETRIC_METHODS]
        
        return BenchmarkResults(
            config=self.config,
            results_by_fraction=results_by_fraction,
            baseline_methods=baseline_methods,
            geometric_methods=geometric_methods,
        )
    
    def _run_trials(self, data: np.ndarray, missing_fraction: float,
                    method: ImputationMethod) -> List[ImputationResult]:
        """Run multiple trials for a single method/fraction combination."""
        results = []
        
        for trial in range(self.config.n_trials):
            # Generate random mask
            seed = self.rng.integers(0, 2**31)
            
            try:
                import jax
                key = jax.random.PRNGKey(seed)
            except ImportError:
                key = seed
            
            # Apply masking
            for pattern in self.config.mask_patterns:
                mask = self._create_mask(data.shape, missing_fraction, seed)
                
                start_time = time.time()
                
                try:
                    imputed = self.benchmark.impute(data, mask, method)
                    
                    # Compute errors
                    missing = ~mask
                    if np.sum(missing) > 0:
                        diff = imputed[missing] - data[missing]
                        rmse = float(np.sqrt(np.mean(diff ** 2)))
                        mae = float(np.mean(np.abs(diff)))
                    else:
                        rmse = 0.0
                        mae = 0.0
                    
                    manifold_error = self.benchmark.compute_manifold_error(data, imputed, mask)
                    is_valid = self.benchmark.validate_on_manifold(imputed)
                    
                    runtime = time.time() - start_time
                    
                    result = ImputationResult(
                        method=method,
                        imputed_data=imputed,
                        original_data=data,
                        mask=mask,
                        rmse=rmse,
                        mae=mae,
                        manifold_error=manifold_error,
                        is_valid=is_valid,
                    )
                    result.runtime = runtime  # Add runtime
                    results.append(result)
                    
                except Exception as e:
                    print(f"    Trial {trial} failed for {method.value}: {e}")
                    continue
        
        return results
    
    def _create_mask(self, shape: tuple, missing_fraction: float, 
                     seed: int) -> np.ndarray:
        """Create a random mask for the data."""
        rng = np.random.default_rng(seed)
        return rng.random(shape) > missing_fraction
    
    def _aggregate_trials(self, method: ImputationMethod,
                          results: List[ImputationResult]) -> MethodResult:
        """Aggregate results across trials."""
        rmses = [r.rmse for r in results]
        maes = [r.mae for r in results]
        manifold_errors = [r.manifold_error for r in results if r.manifold_error is not None]
        mis_scores = [getattr(r, 'mis', 0.0) for r in results]
        validities = [r.is_valid for r in results]
        runtimes = [getattr(r, 'runtime', 0.0) for r in results]
        
        return MethodResult(
            method=method,
            rmse_mean=float(np.mean(rmses)),
            rmse_std=float(np.std(rmses)),
            mae_mean=float(np.mean(maes)),
            mae_std=float(np.std(maes)),
            manifold_error_mean=float(np.mean(manifold_errors)) if manifold_errors else 0.0,
            manifold_error_std=float(np.std(manifold_errors)) if manifold_errors else 0.0,
            mis_mean=float(np.mean(mis_scores)),
            mis_std=float(np.std(mis_scores)),
            validity_rate=float(np.mean(validities)),
            runtime_mean=float(np.mean(runtimes)),
        )
    
    @staticmethod
    def run_quick_benchmark(dataset_name: str = 'physionet_eeg',
                            n_samples: int = 50) -> BenchmarkResults:
        """
        Run a quick benchmark with synthetic/small data for testing.
        """
        config = BenchmarkConfig(
            dataset_name=dataset_name,
            missing_fractions=[0.1, 0.3, 0.5, 0.7, 0.9],
            n_trials=3,
        )
        
        runner = BenchmarkRunner(config)
        
        # Generate appropriate synthetic data
        rng = np.random.default_rng(42)
        
        if dataset_name == 'physionet_eeg':
            # Generate SPD matrices
            dim = 8
            data = []
            for _ in range(n_samples):
                L = rng.standard_normal((dim, dim))
                mat = L @ L.T + 0.1 * np.eye(dim)
                data.append(mat)
            data = np.array(data)
            
        elif dataset_name == 'ghcn_daily':
            # Generate spherical data
            n_points = n_samples
            lat = rng.uniform(-np.pi/2, np.pi/2, n_points)
            lon = rng.uniform(-np.pi, np.pi, n_points)
            temp = 20 - 30 * np.abs(lat) / (np.pi/2) + rng.standard_normal(n_points) * 3
            data = np.column_stack([lat, lon, temp])
            
        elif dataset_name == 'cmu_mocap':
            # Generate joint angle data
            n_frames = n_samples
            n_joints = 10
            t = np.linspace(0, 4*np.pi, n_frames)
            data = np.zeros((n_frames, n_joints, 3))
            for j in range(n_joints):
                data[:, j, 0] = 0.3 * np.sin(t + j)
                data[:, j, 1] = 0.2 * np.cos(t + j)
                data[:, j, 2] = 0.1 * np.sin(2*t + j)
            data += rng.standard_normal(data.shape) * 0.05
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return runner.run(data)

