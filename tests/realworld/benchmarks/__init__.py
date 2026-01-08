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
    SPDTangentSpaceModel,
    FisherImputationModel,
    SPDFisherModel,
    SphericalFisherModel,
    ExtractedFisherModel,
    train_model,
    train_spd_model,
    train_spd_fisher_model,
    train_spherical_fisher_model,
    train_extracted_fisher_model,
    evaluate_model,
    evaluate_spd_model,
    evaluate_spd_fisher_model,
    evaluate_spherical_fisher_model,
    evaluate_extracted_fisher_model,
    run_learned_benchmark,
    LearnedBenchmarkResult,
    # Multi-model infrastructure
    ModelConfig,
    ModelResult,
    MultiBenchmarkResult,
    run_multi_model_benchmark,
    # Common training utilities
    DataSplit,
    prepare_data_for_training,
    create_data_split,
    clip_gradients,
    run_training_loop,
    compute_imputation_metrics,
)

from .learned_viz import (
    plot_learned_benchmark,
    plot_all_metrics,
    save_learned_benchmark_results,
    print_learned_summary,
    # Multi-model visualization
    plot_multi_benchmark,
    plot_multi_metrics,
    save_multi_benchmark_results,
    print_multi_summary,
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
    'SPDTangentSpaceModel',
    'FisherImputationModel',
    'SPDFisherModel',
    'SphericalFisherModel',
    'ExtractedFisherModel',
    'train_model',
    'train_spd_model',
    'train_spd_fisher_model',
    'train_spherical_fisher_model',
    'train_extracted_fisher_model',
    'evaluate_model',
    'evaluate_spd_model',
    'evaluate_spd_fisher_model',
    'evaluate_spherical_fisher_model',
    'evaluate_extracted_fisher_model',
    'run_learned_benchmark',
    'LearnedBenchmarkResult',
    # Multi-model infrastructure
    'ModelConfig',
    'ModelResult',
    'MultiBenchmarkResult',
    'run_multi_model_benchmark',
    # Common training utilities
    'DataSplit',
    'prepare_data_for_training',
    'create_data_split',
    'clip_gradients',
    'run_training_loop',
    'compute_imputation_metrics',
    # Learned visualization
    'plot_learned_benchmark',
    'plot_all_metrics',
    'save_learned_benchmark_results',
    'print_learned_summary',
    # Multi-model visualization
    'plot_multi_benchmark',
    'plot_multi_metrics',
    'save_multi_benchmark_results',
    'print_multi_summary',
    # Manifold Integrity Score
    'ManifoldIntegrityScore',
    'ManifoldIntegrityResult',
    'ManifoldType',
    'compute_mis_for_imputation',
]

