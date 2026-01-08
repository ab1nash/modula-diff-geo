"""
Visualization Utilities for Geometric Imputation Benchmarks

Creates figures comparing baseline (modula/Euclidean) vs diffgeo (geometric) 
imputation performance with clear labeling and multiple metrics.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")


# Enhanced color scheme with clear distinction
COLORS = {
    # Baseline methods (grays/warm colors) - modula/Euclidean
    'zero_fill': '#e74c3c',        # Red - worst baseline
    'mean_fill': '#e67e22',        # Orange
    'linear_interp': '#f39c12',    # Yellow-orange
    'nearest_neighbor': '#95a5a6', # Gray
    
    # Geometric methods (blues/greens) - diffgeo
    'log_euclidean': '#3498db',     # Blue
    'frechet_mean': '#2980b9',      # Darker blue
    'geodesic_interp': '#1abc9c',   # Teal
    'parallel_transport': '#16a085', # Dark teal
    
    # Category colors
    'baseline': '#e74c3c',    # Red for baseline
    'geometric': '#27ae60',   # Green for geometric
}

# Method display names with clear framework labels
METHOD_LABELS = {
    'zero_fill': 'Zero Fill [Baseline]',
    'mean_fill': 'Mean Fill [Baseline]',
    'linear_interp': 'Linear Interp [Baseline]',
    'nearest_neighbor': 'Nearest Neighbor [Baseline]',
    'log_euclidean': 'Log-Euclidean [DiffGeo]',
    'frechet_mean': 'Fréchet Mean [DiffGeo]',
    'geodesic_interp': 'Geodesic Interp [DiffGeo]',
    'parallel_transport': 'Parallel Transport [DiffGeo]',
}

MARKERS = {
    'baseline': 'o',
    'geometric': 's',
}


def _get_method_label(method: str, is_geometric: bool) -> str:
    """Get display label for method with framework tag."""
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    tag = "[DiffGeo]" if is_geometric else "[Baseline]"
    return f"{method.replace('_', ' ').title()} {tag}"


def _compute_relative_improvement(baseline_val: float, geo_val: float) -> float:
    """Compute % improvement of geometric over baseline."""
    if baseline_val == 0:
        return 0.0
    return (baseline_val - geo_val) / baseline_val * 100


def plot_imputation_comparison(results: 'BenchmarkResults',
                                metric: str = 'rmse',
                                save_path: Optional[Path] = None,
                                figsize: Tuple[int, int] = (14, 7)) -> Optional[plt.Figure]:
    """
    Plot comparison of imputation methods across missing fractions.
    
    Clear labeling: [Baseline] vs [DiffGeo] methods.
    """
    if not HAS_MATPLOTLIB:
        _print_text_comparison(results, metric)
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Collect all methods
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    # Separate baseline and geometric methods
    baseline_methods = [m for m in sorted(all_methods) if m in results.baseline_methods]
    geometric_methods = [m for m in sorted(all_methods) if m in results.geometric_methods]
    
    # Plot baseline methods first (dashed lines)
    for method in baseline_methods:
        values = []
        errors = []
        
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                result = results.results_by_fraction[frac][method]
                values.append(getattr(result, f'{metric}_mean', np.nan))
                errors.append(getattr(result, f'{metric}_std', 0))
            else:
                values.append(np.nan)
                errors.append(0)
        
        color = COLORS.get(method, COLORS['baseline'])
        label = _get_method_label(method, False)
        
        ax.errorbar(fractions, values, yerr=errors, 
                   label=label,
                   color=color, marker='o', linestyle='--',
                   capsize=3, markersize=8, linewidth=2, alpha=0.7)
    
    # Plot geometric methods (solid lines, emphasized)
    for method in geometric_methods:
        values = []
        errors = []
        
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                result = results.results_by_fraction[frac][method]
                values.append(getattr(result, f'{metric}_mean', np.nan))
                errors.append(getattr(result, f'{metric}_std', 0))
            else:
                values.append(np.nan)
                errors.append(0)
        
        color = COLORS.get(method, COLORS['geometric'])
        label = _get_method_label(method, True)
        
        ax.errorbar(fractions, values, yerr=errors, 
                   label=label,
                   color=color, marker='s', linestyle='-',
                   capsize=4, markersize=10, linewidth=3)
    
    # Styling
    metric_name = metric.upper().replace('_', ' ')
    ax.set_xlabel('Missing Data Fraction', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    ax.set_title(f'{metric_name} Comparison: {dataset_name}\n'
                 f'Baseline (modula) vs DiffGeo (geometric)', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    # Add annotation box
    textstr = '○-- Baseline (Euclidean)\n■─ DiffGeo (Manifold)'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_multi_metric_comparison(results: 'BenchmarkResults',
                                  save_path: Optional[Path] = None,
                                  figsize: Tuple[int, int] = (16, 10)) -> Optional[plt.Figure]:
    """
    Comprehensive multi-metric comparison figure.
    
    Shows RMSE, MAE, Manifold Error, and Relative Improvement in a 2x2 grid.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Collect methods
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    baseline_methods = [m for m in sorted(all_methods) if m in results.baseline_methods]
    geometric_methods = [m for m in sorted(all_methods) if m in results.geometric_methods]
    
    metrics = [
        ('rmse', 'RMSE (Root Mean Squared Error)', axes[0, 0]),
        ('mae', 'MAE (Mean Absolute Error)', axes[0, 1]),
        ('manifold_error', 'Manifold Distance Error', axes[1, 0]),
    ]
    
    # Plot first 3 metrics
    for metric, title, ax in metrics:
        for method in baseline_methods:
            values = []
            for frac in fractions:
                if method in results.results_by_fraction[frac]:
                    values.append(getattr(results.results_by_fraction[frac][method], f'{metric}_mean', np.nan))
                else:
                    values.append(np.nan)
            
            color = COLORS.get(method, COLORS['baseline'])
            label = method.replace('_', ' ').title()
            ax.plot(fractions, values, 'o--', color=color, label=f'{label} [B]',
                   markersize=6, linewidth=1.5, alpha=0.7)
        
        for method in geometric_methods:
            values = []
            for frac in fractions:
                if method in results.results_by_fraction[frac]:
                    values.append(getattr(results.results_by_fraction[frac][method], f'{metric}_mean', np.nan))
                else:
                    values.append(np.nan)
            
            color = COLORS.get(method, COLORS['geometric'])
            label = method.replace('_', ' ').title()
            ax.plot(fractions, values, 's-', color=color, label=f'{label} [G]',
                   markersize=8, linewidth=2.5)
        
        ax.set_xlabel('Missing Fraction', fontsize=10)
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    # Plot 4: Relative improvement (%)
    ax = axes[1, 1]
    
    improvements = {frac: [] for frac in fractions}
    
    for frac in fractions:
        # Get best baseline RMSE
        best_baseline = float('inf')
        for method in baseline_methods:
            if method in results.results_by_fraction[frac]:
                val = results.results_by_fraction[frac][method].rmse_mean
                best_baseline = min(best_baseline, val)
        
        # Get best geometric RMSE
        best_geo = float('inf')
        for method in geometric_methods:
            if method in results.results_by_fraction[frac]:
                val = results.results_by_fraction[frac][method].rmse_mean
                best_geo = min(best_geo, val)
        
        if best_baseline != float('inf') and best_geo != float('inf'):
            improvement = _compute_relative_improvement(best_baseline, best_geo)
            improvements[frac] = improvement
    
    bars = ax.bar([f'{f:.0%}' for f in fractions], 
                  [improvements.get(f, 0) for f in fractions],
                  color=[COLORS['geometric'] if improvements.get(f, 0) > 0 else COLORS['baseline'] 
                         for f in fractions],
                  edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Missing Fraction', fontsize=10)
    ax.set_ylabel('Improvement (%)', fontsize=10)
    ax.set_title('DiffGeo vs Baseline: RMSE Improvement %\n(positive = DiffGeo better)', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, frac in zip(bars, fractions):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    fig.suptitle(f'Comprehensive Imputation Benchmark: {dataset_name}\n'
                 f'[B] = Baseline (modula/Euclidean)  |  [G] = DiffGeo (Geometric)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_degradation_curves(results: 'BenchmarkResults',
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> Optional[plt.Figure]:
    """
    Plot how different metrics degrade with increasing missing fraction.
    
    Creates a 3-panel figure showing RMSE, MAE, and manifold error.
    Clearly shows Best Baseline vs Best DiffGeo.
    """
    if not HAS_MATPLOTLIB:
        _print_text_degradation(results)
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = ['rmse', 'mae', 'manifold_error']
    titles = ['RMSE', 'MAE', 'Manifold Error']
    
    fractions = sorted(results.results_by_fraction.keys())
    
    for ax, metric, title in zip(axes, metrics, titles):
        baseline_values = []
        geometric_values = []
        baseline_names = []
        geometric_names = []
        
        for frac in fractions:
            baseline_best = float('inf')
            geometric_best = float('inf')
            best_base_name = ""
            best_geo_name = ""
            
            for method, result in results.results_by_fraction[frac].items():
                value = getattr(result, f'{metric}_mean', float('inf'))
                
                if method in results.baseline_methods:
                    if value < baseline_best:
                        baseline_best = value
                        best_base_name = method
                elif method in results.geometric_methods:
                    if value < geometric_best:
                        geometric_best = value
                        best_geo_name = method
            
            baseline_values.append(baseline_best if baseline_best != float('inf') else np.nan)
            geometric_values.append(geometric_best if geometric_best != float('inf') else np.nan)
            baseline_names.append(best_base_name)
            geometric_names.append(best_geo_name)
        
        # Plot with clear labels
        ax.plot(fractions, baseline_values, 'o--', color=COLORS['baseline'],
               label='Best Baseline [modula]', linewidth=2.5, markersize=10)
        ax.plot(fractions, geometric_values, 's-', color=COLORS['geometric'],
               label='Best DiffGeo [geometric]', linewidth=3, markersize=12)
        
        # Highlight winner at each fraction
        for i, (frac, base, geo) in enumerate(zip(fractions, baseline_values, geometric_values)):
            if not np.isnan(base) and not np.isnan(geo):
                winner_color = COLORS['geometric'] if geo < base else COLORS['baseline']
                ax.axvspan(frac - 0.025, frac + 0.025, alpha=0.15, color=winner_color)
        
        ax.set_xlabel('Missing Fraction', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Degradation', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    fig.suptitle(f'Performance Degradation Curves: {dataset_name}\n'
                 f'Shading indicates winner at each missing fraction',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_manifold_distances(results: 'BenchmarkResults',
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
    """
    Plot manifold-specific error comparison as heatmap with clear labels.
    """
    if not HAS_MATPLOTLIB:
        _print_text_heatmap(results)
        return None
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Collect all methods
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    # Sort: baseline first, then geometric
    baseline = [m for m in sorted(all_methods) if m in results.baseline_methods]
    geometric = [m for m in sorted(all_methods) if m in results.geometric_methods]
    methods = baseline + geometric
    
    # Build matrix
    data = np.zeros((len(methods), len(fractions)))
    
    for i, method in enumerate(methods):
        for j, frac in enumerate(fractions):
            if method in results.results_by_fraction[frac]:
                data[i, j] = results.results_by_fraction[frac][method].manifold_error_mean
            else:
                data[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r')
    
    # Create labels with framework tags
    y_labels = []
    for method in methods:
        is_geo = method in results.geometric_methods
        label = _get_method_label(method, is_geo)
        y_labels.append(label)
    
    # Labels
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f'{f:.0%}' for f in fractions], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    ax.set_xlabel('Missing Fraction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    ax.set_title(f'Manifold Error Heatmap: {dataset_name}\n'
                 f'(Lower is better)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Manifold Error', fontsize=11)
    
    # Add values as text
    for i in range(len(methods)):
        for j in range(len(fractions)):
            if not np.isnan(data[i, j]):
                text_color = 'white' if data[i, j] > np.nanmean(data) else 'black'
                ax.text(j, i, f'{data[i, j]:.3f}',
                       ha='center', va='center', fontsize=9, color=text_color,
                       fontweight='bold')
    
    # Color code row labels
    for i, method in enumerate(methods):
        is_geo = method in results.geometric_methods
        color = COLORS['geometric'] if is_geo else COLORS['baseline']
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight('bold')
    
    # Add separator line between baseline and geometric
    if baseline and geometric:
        ax.axhline(y=len(baseline) - 0.5, color='black', linewidth=2, linestyle='-')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_manifold_integrity(results: 'BenchmarkResults',
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = (12, 7)) -> Optional[plt.Figure]:
    """
    Plot Manifold Integrity Score (MIS) comparison.
    
    MIS is a continuous metric (0 = perfect manifold membership, higher = worse).
    This replaces the binary validity rate with a more informative measure.
    
    See docs/manifold_integrity_score.md for full documentation.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Collect all methods
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    baseline_methods = [m for m in sorted(all_methods) if m in results.baseline_methods]
    geometric_methods = [m for m in sorted(all_methods) if m in results.geometric_methods]
    
    x = np.arange(len(fractions))
    n_methods = len(baseline_methods) + len(geometric_methods)
    width = 0.7 / n_methods
    
    # Plot baseline methods
    for i, method in enumerate(baseline_methods):
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(results.results_by_fraction[frac][method].mis_mean)
            else:
                values.append(0)
        
        color = COLORS.get(method, COLORS['baseline'])
        label = _get_method_label(method, False)
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=label,
              color=color, alpha=0.7, edgecolor='black')
    
    # Plot geometric methods
    for i, method in enumerate(geometric_methods):
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(results.results_by_fraction[frac][method].mis_mean)
            else:
                values.append(0)
        
        color = COLORS.get(method, COLORS['geometric'])
        label = _get_method_label(method, True)
        offset = (len(baseline_methods) + i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=label,
              color=color, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Missing Fraction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Manifold Integrity Score (MIS)', fontsize=12, fontweight='bold')
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    ax.set_title(f'Manifold Integrity Score: {dataset_name}\n'
                 f'(0 = perfect manifold membership, lower is better)', 
                 fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    # Add threshold line (0.01 = valid)
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, linewidth=2,
               label='Valid threshold (0.01)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


# Keep old name for backwards compatibility
def plot_validity_comparison(results: 'BenchmarkResults',
                             save_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = (12, 7)) -> Optional[plt.Figure]:
    """
    DEPRECATED: Use plot_manifold_integrity() instead.
    This function now plots MIS instead of binary validity rate.
    """
    return plot_manifold_integrity(results, save_path, figsize)


def plot_summary_table(results: 'BenchmarkResults',
                       save_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (14, 8)) -> Optional[plt.Figure]:
    """
    Create a summary table figure with all metrics.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Build table data
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    methods = sorted(all_methods)
    
    # Headers
    headers = ['Method', 'Type'] + [f'{f:.0%} Missing' for f in fractions] + ['Avg RMSE', 'Avg MAE']
    
    # Data rows
    table_data = []
    for method in methods:
        is_geo = method in results.geometric_methods
        row = [
            method.replace('_', ' ').title(),
            'DiffGeo' if is_geo else 'Baseline'
        ]
        
        rmse_values = []
        mae_values = []
        
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                result = results.results_by_fraction[frac][method]
                row.append(f'{result.rmse_mean:.2f}')
                rmse_values.append(result.rmse_mean)
                mae_values.append(result.mae_mean)
            else:
                row.append('N/A')
        
        row.append(f'{np.mean(rmse_values):.2f}' if rmse_values else 'N/A')
        row.append(f'{np.mean(mae_values):.2f}' if mae_values else 'N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code rows
    for i, method in enumerate(methods):
        is_geo = method in results.geometric_methods
        color = '#d5f5e3' if is_geo else '#fadbd8'  # Light green vs light red
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    ax.set_title(f'Summary Table: {dataset_name}\n'
                 f'Green = DiffGeo (Geometric)  |  Red = Baseline (Euclidean)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_comprehensive_benchmark(results: 'BenchmarkResults',
                                  save_path: Optional[Path] = None,
                                  figsize: Tuple[int, int] = (18, 16)) -> Optional[plt.Figure]:
    """
    Create a single comprehensive benchmark figure with all key metrics.
    
    Layout (3x2):
    - Top left: RMSE comparison
    - Top right: MAE comparison  
    - Middle left: MIS (Manifold Integrity Score)
    - Middle right: Relative improvement bars
    - Bottom: Summary statistics table with MIS
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.25)
    
    fractions = sorted(results.results_by_fraction.keys())
    
    # Collect methods
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    baseline_methods = [m for m in sorted(all_methods) if m in results.baseline_methods]
    geometric_methods = [m for m in sorted(all_methods) if m in results.geometric_methods]
    
    # ===== TOP LEFT: RMSE Comparison =====
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_metric_panel(ax1, results, fractions, baseline_methods, geometric_methods, 'rmse', 'RMSE')
    
    # ===== TOP RIGHT: MAE Comparison =====
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_metric_panel(ax2, results, fractions, baseline_methods, geometric_methods, 'mae', 'MAE')
    
    # ===== MIDDLE LEFT: MIS (Manifold Integrity Score) =====
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_mis_panel(ax3, results, fractions, baseline_methods, geometric_methods)
    
    # ===== MIDDLE RIGHT: Improvement Bars =====
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_improvement_bars(ax4, results, fractions, baseline_methods, geometric_methods)
    
    # ===== BOTTOM: Summary Stats =====
    ax5 = fig.add_subplot(gs[2, :])
    _plot_summary_stats(ax5, results, fractions, baseline_methods, geometric_methods)
    
    # Main title
    dataset_name = results.config.dataset_name.replace('_', ' ').title()
    fig.suptitle(f'Geometric Imputation Benchmark: {dataset_name}\n'
                 f'○-- Baseline (Euclidean/modula)  |  ■─ DiffGeo (Geometric)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def _plot_metric_panel(ax, results, fractions, baseline_methods, geometric_methods, metric, title):
    """Plot a single metric comparison panel."""
    for method in baseline_methods:
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(getattr(results.results_by_fraction[frac][method], f'{metric}_mean', np.nan))
            else:
                values.append(np.nan)
        
        color = COLORS.get(method, COLORS['baseline'])
        label = method.replace('_', ' ').title()
        ax.plot(fractions, values, 'o--', color=color, label=f'{label} [B]',
               markersize=7, linewidth=1.5, alpha=0.7)
    
    for method in geometric_methods:
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(getattr(results.results_by_fraction[frac][method], f'{metric}_mean', np.nan))
            else:
                values.append(np.nan)
        
        color = COLORS.get(method, COLORS['geometric'])
        label = method.replace('_', ' ').title()
        ax.plot(fractions, values, 's-', color=color, label=f'{label} [G]',
               markersize=9, linewidth=2.5)
    
    ax.set_xlabel('Missing Fraction', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(f'{title} (lower is better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])


def _plot_mis_panel(ax, results, fractions, baseline_methods, geometric_methods):
    """
    Plot Manifold Integrity Score (MIS) panel.
    
    MIS is our custom metric that measures how well data respects manifold constraints.
    0 = perfect, higher = worse. See docs/manifold_integrity_score.md
    """
    x = np.arange(len(fractions))
    n_methods = len(baseline_methods) + len(geometric_methods)
    width = 0.7 / max(n_methods, 1)
    
    # Plot baseline methods as bars
    for i, method in enumerate(baseline_methods):
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(results.results_by_fraction[frac][method].mis_mean)
            else:
                values.append(0)
        
        color = COLORS.get(method, COLORS['baseline'])
        label = method.replace('_', ' ').title()
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.85, label=f'{label} [B]',
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Plot geometric methods as bars
    for i, method in enumerate(geometric_methods):
        values = []
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                values.append(results.results_by_fraction[frac][method].mis_mean)
            else:
                values.append(0)
        
        color = COLORS.get(method, COLORS['geometric'])
        label = method.replace('_', ' ').title()
        offset = (len(baseline_methods) + i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.85, label=f'{label} [G]',
               color=color, edgecolor='black', linewidth=1.2)
    
    # Add validity threshold line
    ax.axhline(y=0.01, color='green', linestyle='--', linewidth=2, alpha=0.8,
               label='Valid (MIS < 0.01)')
    
    ax.set_xlabel('Missing Fraction', fontsize=10)
    ax.set_ylabel('Manifold Integrity Score', fontsize=10)
    ax.set_title('MIS - Manifold Integrity Score (lower is better)\n'
                 '0 = perfect manifold membership', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])


def _plot_improvement_bars(ax, results, fractions, baseline_methods, geometric_methods):
    """Plot relative improvement bars."""
    improvements = []
    
    for frac in fractions:
        best_baseline = float('inf')
        best_geo = float('inf')
        
        for method in baseline_methods:
            if method in results.results_by_fraction[frac]:
                val = results.results_by_fraction[frac][method].rmse_mean
                best_baseline = min(best_baseline, val)
        
        for method in geometric_methods:
            if method in results.results_by_fraction[frac]:
                val = results.results_by_fraction[frac][method].rmse_mean
                best_geo = min(best_geo, val)
        
        if best_baseline != float('inf') and best_geo != float('inf') and best_baseline > 0:
            improvement = (best_baseline - best_geo) / best_baseline * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors = [COLORS['geometric'] if imp > 0 else COLORS['baseline'] for imp in improvements]
    bars = ax.bar([f'{f:.0%}' for f in fractions], improvements, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Missing Fraction', fontsize=10)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=10)
    ax.set_title('DiffGeo vs Baseline Improvement\n(+ve = DiffGeo wins)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:+.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')


def _plot_summary_stats(ax, results, fractions, baseline_methods, geometric_methods):
    """Plot summary statistics as a table including MIS."""
    ax.axis('off')
    
    # Calculate summary stats
    rows = []
    
    for method in baseline_methods + geometric_methods:
        is_geo = method in geometric_methods
        rmse_vals = []
        mae_vals = []
        mis_vals = []
        
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                r = results.results_by_fraction[frac][method]
                rmse_vals.append(r.rmse_mean)
                mae_vals.append(r.mae_mean)
                mis_vals.append(r.mis_mean)
        
        if rmse_vals:
            avg_mis = np.mean(mis_vals)
            # Flag if MIS indicates invalid results
            mis_str = f'{avg_mis:.4f}'
            if avg_mis > 0.01:
                mis_str += ' ⚠'  # Warning if above valid threshold
            
            rows.append([
                method.replace('_', ' ').title(),
                'DiffGeo' if is_geo else 'Baseline',
                f'{np.mean(rmse_vals):.3f}',
                f'{np.mean(mae_vals):.3f}',
                mis_str,
            ])
    
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=['Method', 'Type', 'Avg RMSE', 'Avg MAE', 'Avg MIS'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
        
        # Color rows
        for i, method in enumerate(baseline_methods + geometric_methods):
            is_geo = method in geometric_methods
            color = '#d5f5e3' if is_geo else '#fadbd8'
            for j in range(5):
                table[(i + 1, j)].set_facecolor(color)
        
        # Header
        for j in range(5):
            table[(0, j)].set_facecolor('#2c3e50')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Add MIS explanation note
    ax.text(0.5, -0.1, 'MIS = Manifold Integrity Score (0 = perfect, >0.01 = invalid, ⚠ indicates violation)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')
    
    ax.set_title('Summary Statistics (averaged across all missing fractions)', 
                 fontsize=11, fontweight='bold', y=0.95)


def save_benchmark_figures(results: 'BenchmarkResults',
                           output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate and save benchmark figures - consolidated into 2 key figures.
    """
    if output_dir is None:
        output_dir = results.config.output_dir / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    dataset = results.config.dataset_name
    timestamp = results.timestamp
    
    # Generate only 2 consolidated figures
    figures = [
        ('benchmark', lambda: plot_comprehensive_benchmark(results)),
        ('degradation', lambda: plot_degradation_curves(results)),
    ]
    
    for name, fig_func in figures:
        try:
            # Timestamp prefix for easy sorting
            save_path = output_dir / f"{timestamp}_{dataset}_{name}.png"
            fig = fig_func()
            if fig is not None:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved[name] = save_path
                print(f"  Saved: {save_path.name}")
        except Exception as e:
            print(f"  Warning: Failed to generate {name}: {e}")
    
    return saved


# Fallback text-based outputs

def _print_text_comparison(results: 'BenchmarkResults', metric: str):
    """Print text comparison."""
    print(f"\n{'='*70}")
    print(f"Imputation Comparison ({metric.upper()})")
    print(f"{'='*70}")
    
    fractions = sorted(results.results_by_fraction.keys())
    
    print(f"{'Method':<25} {'Type':<10}", end='')
    for frac in fractions:
        print(f"{frac:>10.0%}", end='')
    print()
    print("-" * (35 + 10 * len(fractions)))
    
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    for method in sorted(all_methods):
        is_geo = method in results.geometric_methods
        type_label = "DiffGeo" if is_geo else "Baseline"
        marker = "◆" if is_geo else "○"
        
        print(f"{marker} {method:<23} {type_label:<10}", end='')
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                result = results.results_by_fraction[frac][method]
                value = getattr(result, f'{metric}_mean', 0)
                print(f"{value:>10.4f}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()


def _print_text_degradation(results: 'BenchmarkResults'):
    """Print text degradation summary."""
    print(f"\n{'='*70}")
    print("Degradation Summary: Best Baseline vs Best DiffGeo")
    print(f"{'='*70}")
    
    fractions = sorted(results.results_by_fraction.keys())
    
    print(f"{'Fraction':<10} {'Best Baseline':<20} {'RMSE':<10} {'Best DiffGeo':<20} {'RMSE':<10} {'Winner':<10}")
    print("-" * 80)
    
    for frac in fractions:
        best_base_name = ""
        best_base_rmse = float('inf')
        best_geo_name = ""
        best_geo_rmse = float('inf')
        
        for method, result in results.results_by_fraction[frac].items():
            if method in results.baseline_methods:
                if result.rmse_mean < best_base_rmse:
                    best_base_rmse = result.rmse_mean
                    best_base_name = method
            elif method in results.geometric_methods:
                if result.rmse_mean < best_geo_rmse:
                    best_geo_rmse = result.rmse_mean
                    best_geo_name = method
        
        winner = "DiffGeo ◆" if best_geo_rmse < best_base_rmse else "Baseline ○"
        
        print(f"{frac:<10.0%} {best_base_name:<20} {best_base_rmse:<10.4f} "
              f"{best_geo_name:<20} {best_geo_rmse:<10.4f} {winner:<10}")


def _print_text_heatmap(results: 'BenchmarkResults'):
    """Print text heatmap."""
    print(f"\n{'='*70}")
    print("Manifold Error Matrix")
    print(f"{'='*70}")
    
    fractions = sorted(results.results_by_fraction.keys())
    
    print(f"{'Method':<25} {'Type':<10}", end='')
    for frac in fractions:
        print(f"{frac:>10.0%}", end='')
    print()
    print("-" * (35 + 10 * len(fractions)))
    
    all_methods = set()
    for frac_results in results.results_by_fraction.values():
        all_methods.update(frac_results.keys())
    
    for method in sorted(all_methods):
        is_geo = method in results.geometric_methods
        type_label = "DiffGeo" if is_geo else "Baseline"
        marker = "◆" if is_geo else "○"
        
        print(f"{marker} {method:<23} {type_label:<10}", end='')
        for frac in fractions:
            if method in results.results_by_fraction[frac]:
                result = results.results_by_fraction[frac][method]
                print(f"{result.manifold_error_mean:>10.4f}", end='')
            else:
                print(f"{'N/A':>10}", end='')
        print()
