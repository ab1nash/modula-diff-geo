"""
Benchmark Framework for Geometric Imputation

Compares modula (baseline) vs diffgeo (geometric) approaches
on real-world datasets.

Two types of benchmarks:
1. Closed-form methods (imputation.py) - analytical comparison
2. Learned models (learnable.py) - trained with proper validation
"""
from .imputation import (
    ImputationBenchmark,
    ImputationMethod,
    SPDImputationMethods,
    SphericalImputationMethods,
    MocapImputationMethods,
)

from .runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResults,
)

from .visualization import (
    plot_imputation_comparison,
    plot_degradation_curves,
    plot_manifold_distances,
    plot_comprehensive_benchmark,
    save_benchmark_figures,
)

from .learnable import (
    TrainingConfig,
    TrainingHistory,
    ImputationModel,
    ModulaImputationModel,
    DiffGeoImputationModel,
    SPDImputationModel,
    train_model,
    evaluate_model,
    run_learned_benchmark,
    LearnedBenchmarkResult,
)

from .learned_viz import (
    plot_learned_benchmark,
    plot_all_metrics,
    save_learned_benchmark_results,
    print_learned_summary,
)

from .manifold_integrity import (
    ManifoldIntegrityScore,
    ManifoldIntegrityResult,
    ManifoldType,
    compute_mis_for_imputation,
)

__all__ = [
    # Closed-form imputation
    'ImputationBenchmark',
    'ImputationMethod',
    'SPDImputationMethods',
    'SphericalImputationMethods',
    'MocapImputationMethods',
    # Runner
    'BenchmarkRunner',
    'BenchmarkConfig',
    'BenchmarkResults',
    # Visualization
    'plot_imputation_comparison',
    'plot_degradation_curves',
    'plot_manifold_distances',
    'plot_comprehensive_benchmark',
    'save_benchmark_figures',
    # Learned models
    'TrainingConfig',
    'TrainingHistory',
    'ImputationModel',
    'ModulaImputationModel',
    'DiffGeoImputationModel',
    'SPDImputationModel',
    'train_model',
    'evaluate_model',
    'run_learned_benchmark',
    'LearnedBenchmarkResult',
    # Learned visualization
    'plot_learned_benchmark',
    'plot_all_metrics',
    'save_learned_benchmark_results',
    'print_learned_summary',
    # Manifold Integrity Score
    'ManifoldIntegrityScore',
    'ManifoldIntegrityResult',
    'ManifoldType',
    'compute_mis_for_imputation',
]

