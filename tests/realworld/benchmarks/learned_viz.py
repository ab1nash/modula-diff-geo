"""
Visualization for Learned Imputation Model Benchmarks

Creates figures showing training curves and comparison of
Modula (Euclidean) vs DiffGeo (Riemannian) learned imputation.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


COLORS = {
    'modula': '#e74c3c',      # Red for baseline
    'diffgeo': '#27ae60',     # Green for geometric
}


def plot_learned_benchmark(results: List['LearnedBenchmarkResult'],
                           save_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = (16, 12)) -> Optional[plt.Figure]:
    """
    Create comprehensive figure for learned model benchmark.
    
    Layout (2x2):
    - Top left: Training curves (loss vs epoch)
    - Top right: Final metrics comparison (bar chart)
    - Bottom left: Convergence speed comparison
    - Bottom right: Improvement summary
    """
    if not HAS_MATPLOTLIB or not results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    dataset_name = results[0].dataset_name.replace('_', ' ').title()
    timestamp = results[0].timestamp
    
    # ===== TOP LEFT: Training Curves =====
    ax1 = axes[0, 0]
    
    # Use the middle missing fraction for training curves
    mid_idx = len(results) // 2
    mid_result = results[mid_idx]
    
    modula_losses = mid_result.modula_history.get('val_losses', [])
    diffgeo_losses = mid_result.diffgeo_history.get('val_losses', [])
    
    if modula_losses and diffgeo_losses:
        epochs_m = range(len(modula_losses))
        epochs_g = range(len(diffgeo_losses))
        
        ax1.plot(epochs_m, modula_losses, '--', color=COLORS['modula'], 
                label='Modula [Baseline]', linewidth=2, alpha=0.8)
        ax1.plot(epochs_g, diffgeo_losses, '-', color=COLORS['diffgeo'],
                label='DiffGeo [Geometric]', linewidth=2.5)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Validation Loss', fontsize=11)
        ax1.set_title(f'Training Curves ({mid_result.missing_fraction:.0%} missing)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # ===== TOP RIGHT: Final Metrics Comparison =====
    ax2 = axes[0, 1]
    
    fractions = [r.missing_fraction for r in results]
    modula_rmse = [r.modula_metrics.get('rmse', 0) for r in results]
    diffgeo_rmse = [r.diffgeo_metrics.get('rmse', 0) for r in results]
    
    # Get std if available (multi-run benchmarks)
    modula_rmse_std = [getattr(r, 'modula_metrics_std', {}).get('rmse', 0) for r in results]
    diffgeo_rmse_std = [getattr(r, 'diffgeo_metrics_std', {}).get('rmse', 0) for r in results]
    has_std = any(s > 0 for s in modula_rmse_std + diffgeo_rmse_std)
    
    x = np.arange(len(fractions))
    width = 0.35
    
    if has_std:
        # Bar chart with error bars
        bars1 = ax2.bar(x - width/2, modula_rmse, width, label='Modula [Baseline]',
                       color=COLORS['modula'], alpha=0.8, edgecolor='black',
                       yerr=modula_rmse_std, capsize=3)
        bars2 = ax2.bar(x + width/2, diffgeo_rmse, width, label='DiffGeo [Geometric]',
                       color=COLORS['diffgeo'], edgecolor='black',
                       yerr=diffgeo_rmse_std, capsize=3)
    else:
        bars1 = ax2.bar(x - width/2, modula_rmse, width, label='Modula [Baseline]',
                       color=COLORS['modula'], alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x + width/2, diffgeo_rmse, width, label='DiffGeo [Geometric]',
                       color=COLORS['diffgeo'], edgecolor='black')
    
    ax2.set_xlabel('Missing Fraction', fontsize=11)
    ax2.set_ylabel('RMSE (lower is better)', fontsize=11)
    n_runs = getattr(results[0], 'n_runs', 1)
    title_suffix = f' (n={n_runs} runs)' if n_runs > 1 else ''
    ax2.set_title(f'Final RMSE by Missing Fraction{title_suffix}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{f:.0%}' for f in fractions])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels (show mean ± std if available)
    for i, bar in enumerate(bars1):
        if has_std and modula_rmse_std[i] > 0:
            label = f'{modula_rmse[i]:.2f}±{modula_rmse_std[i]:.2f}'
        else:
            label = f'{bar.get_height():.3f}'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=7)
    for i, bar in enumerate(bars2):
        if has_std and diffgeo_rmse_std[i] > 0:
            label = f'{diffgeo_rmse[i]:.2f}±{diffgeo_rmse_std[i]:.2f}'
        else:
            label = f'{bar.get_height():.3f}'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=7)
    
    # ===== BOTTOM LEFT: Convergence Speed =====
    ax3 = axes[1, 0]
    
    modula_epochs = [r.modula_history.get('best_epoch', 0) for r in results]
    diffgeo_epochs = [r.diffgeo_history.get('best_epoch', 0) for r in results]
    
    ax3.plot(fractions, modula_epochs, 'o--', color=COLORS['modula'],
            label='Modula [Baseline]', markersize=10, linewidth=2)
    ax3.plot(fractions, diffgeo_epochs, 's-', color=COLORS['diffgeo'],
            label='DiffGeo [Geometric]', markersize=12, linewidth=2.5)
    
    ax3.set_xlabel('Missing Fraction', fontsize=11)
    ax3.set_ylabel('Epochs to Best Validation', fontsize=11)
    ax3.set_title('Convergence Speed (lower = faster)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(fractions)
    ax3.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    # ===== BOTTOM RIGHT: Improvement Summary =====
    ax4 = axes[1, 1]
    
    improvements = []
    for r in results:
        modula_rmse = r.modula_metrics.get('rmse', 0)
        diffgeo_rmse = r.diffgeo_metrics.get('rmse', 0)
        if modula_rmse > 0:
            imp = (modula_rmse - diffgeo_rmse) / modula_rmse * 100
        else:
            imp = 0
        improvements.append(imp)
    
    colors = [COLORS['diffgeo'] if imp > 0 else COLORS['modula'] for imp in improvements]
    bars = ax4.bar([f'{f:.0%}' for f in fractions], improvements, color=colors,
                  edgecolor='black', linewidth=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Missing Fraction', fontsize=11)
    ax4.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax4.set_title('DiffGeo vs Modula Improvement\n(+ve = DiffGeo wins)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax4.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if imp >= 0 else -15),
                    textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')
    
    # Main title
    fig.suptitle(f'Learned Imputation Benchmark: {dataset_name}\n'
                 f'Modula (Euclidean) vs DiffGeo (Riemannian) — {timestamp}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_all_metrics(results: List['LearnedBenchmarkResult'],
                     save_path: Optional[Path] = None,
                     figsize: Tuple[int, int] = (14, 10)) -> Optional[plt.Figure]:
    """
    Plot all metrics (RMSE, MAE, R², MIS) comparison with error bands if available.
    """
    if not HAS_MATPLOTLIB or not results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    fractions = [r.missing_fraction for r in results]
    metrics = ['rmse', 'mae', 'r2', 'mis']
    titles = ['RMSE (↓)', 'MAE (↓)', 'R² (↑)', 'MIS (↓)']
    
    # Check if we have std values
    has_std = hasattr(results[0], 'modula_metrics_std') and results[0].modula_metrics_std
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        modula_vals = np.array([r.modula_metrics.get(metric, 0) for r in results])
        diffgeo_vals = np.array([r.diffgeo_metrics.get(metric, 0) for r in results])
        
        if has_std:
            modula_std = np.array([getattr(r, 'modula_metrics_std', {}).get(metric, 0) for r in results])
            diffgeo_std = np.array([getattr(r, 'diffgeo_metrics_std', {}).get(metric, 0) for r in results])
            
            # Plot with error bands
            ax.fill_between(fractions, modula_vals - modula_std, modula_vals + modula_std,
                           color=COLORS['modula'], alpha=0.2)
            ax.fill_between(fractions, diffgeo_vals - diffgeo_std, diffgeo_vals + diffgeo_std,
                           color=COLORS['diffgeo'], alpha=0.2)
        
        ax.plot(fractions, modula_vals, 'o--', color=COLORS['modula'],
               label='Modula [Baseline]', markersize=10, linewidth=2)
        ax.plot(fractions, diffgeo_vals, 's-', color=COLORS['diffgeo'],
               label='DiffGeo [Geometric]', markersize=12, linewidth=2.5)
        
        ax.set_xlabel('Missing Fraction', fontsize=10)
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fractions)
        ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    dataset_name = results[0].dataset_name.replace('_', ' ').title()
    timestamp = results[0].timestamp
    n_runs = getattr(results[0], 'n_runs', 1)
    
    run_info = f' (n={n_runs} runs, showing mean±std)' if n_runs > 1 else ''
    fig.suptitle(f'All Metrics Comparison: {dataset_name}{run_info}\n'
                 f'Timestamp: {timestamp}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def save_learned_benchmark_results(results: List['LearnedBenchmarkResult'],
                                    output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Save learned benchmark results and figures with timestamp prefix.
    """
    if not results:
        return {}
    
    timestamp = results[0].timestamp
    dataset = results[0].dataset_name
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Save JSON results in json/ subdirectory
    json_dir = output_dir / "json"
    json_dir.mkdir(exist_ok=True)
    json_path = json_dir / f"{timestamp}_{dataset}_learned_benchmark.json"
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    saved['json'] = json_path
    print(f"  Saved: json/{json_path.name}")
    
    # Save figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Main benchmark figure
    fig_path = fig_dir / f"{timestamp}_{dataset}_learned_benchmark.png"
    fig = plot_learned_benchmark(results, fig_path)
    if fig:
        plt.close(fig)
        saved['benchmark'] = fig_path
    
    # All metrics figure
    metrics_path = fig_dir / f"{timestamp}_{dataset}_learned_metrics.png"
    fig = plot_all_metrics(results, metrics_path)
    if fig:
        plt.close(fig)
        saved['metrics'] = metrics_path
    
    return saved


def _get_convergence_flag(final_epoch: int, best_epoch: int, converged: bool) -> str:
    """
    Get flag indicating convergence quality.
    
    Flags:
        ⚡ - Fast convergence (< 100 epochs)
        🎯 - Optimal (best epoch near final, good convergence)
        ⚠️  - Slow convergence (used >80% of epochs)
        🔴 - Did not converge (hit max epochs without early stop)
    """
    if final_epoch == 0:
        return ""
    
    # Fast convergence - found best early
    if best_epoch < 100 and converged:
        return "⚡"
    
    # Optimal - best near end, properly converged
    if converged and (final_epoch - best_epoch) < 50:
        return "🎯"
    
    # No convergence flag
    if not converged:
        return "🔴"
    
    # Slow - took many epochs
    if best_epoch > final_epoch * 0.7:
        return "⚠️"
    
    return ""


def print_learned_summary(results: List['LearnedBenchmarkResult']):
    """Print summary of learned benchmark results."""
    if not results:
        print("No results to summarize")
        return
    
    n_runs = getattr(results[0], 'n_runs', 1)
    has_std = n_runs > 1
    
    print("\n" + "=" * 80)
    print(f"LEARNED IMPUTATION BENCHMARK SUMMARY")
    print(f"Dataset: {results[0].dataset_name}")
    print(f"Timestamp: {results[0].timestamp}")
    if has_std:
        print(f"Runs per condition: {n_runs} (showing mean ± std)")
    print("=" * 80)
    
    # Performance table
    if has_std:
        print(f"\n{'Missing':<10} {'Modula RMSE':<20} {'DiffGeo RMSE':<20} {'Improvement':<12} {'Winner':<10}")
        print("-" * 76)
    else:
        print(f"\n{'Missing':<10} {'Modula RMSE':<13} {'DiffGeo RMSE':<14} {'Improvement':<12} {'Winner':<10}")
        print("-" * 62)
    
    total_improvement = 0
    diffgeo_wins = 0
    
    for r in results:
        modula_rmse = r.modula_metrics.get('rmse', 0)
        diffgeo_rmse = r.diffgeo_metrics.get('rmse', 0)
        
        if modula_rmse > 0:
            improvement = (modula_rmse - diffgeo_rmse) / modula_rmse * 100
        else:
            improvement = 0
        
        total_improvement += improvement
        
        winner = "DiffGeo ◆" if diffgeo_rmse < modula_rmse else "Modula ○"
        if diffgeo_rmse < modula_rmse:
            diffgeo_wins += 1
        
        if has_std:
            modula_std = getattr(r, 'modula_metrics_std', {}).get('rmse', 0)
            diffgeo_std = getattr(r, 'diffgeo_metrics_std', {}).get('rmse', 0)
            print(f"{r.missing_fraction:<10.0%} {modula_rmse:.4f}±{modula_std:.4f}  "
                  f"{diffgeo_rmse:.4f}±{diffgeo_std:.4f}  "
                  f"{improvement:+11.1f}% {winner:<10}")
        else:
            print(f"{r.missing_fraction:<10.0%} {modula_rmse:<13.4f} {diffgeo_rmse:<14.4f} "
                  f"{improvement:+11.1f}% {winner:<10}")
    
    avg_improvement = total_improvement / len(results)
    
    print("-" * (76 if has_std else 62))
    
    # Training details table
    print(f"\n{'Training Details':^90}")
    print("-" * 90)
    print(f"{'Missing':<10} {'Modula':<40} {'DiffGeo':<40}")
    print(f"{'':10} {'Epochs':>10} {'Best@':>10} {'Time(s)':>10} {'':>6} "
          f"{'Epochs':>10} {'Best@':>10} {'Time(s)':>10} {'':>6}")
    print("-" * 90)
    
    for r in results:
        m_hist = r.modula_history
        d_hist = r.diffgeo_history
        
        m_epochs = m_hist.get('final_epoch', len(m_hist.get('train_losses', [])))
        m_best = m_hist.get('best_epoch', 0)
        m_time = r.modula_training_time
        m_converged = m_hist.get('converged', False)
        
        d_epochs = d_hist.get('final_epoch', len(d_hist.get('train_losses', [])))
        d_best = d_hist.get('best_epoch', 0)
        d_time = r.diffgeo_training_time
        d_converged = d_hist.get('converged', False)
        
        # Highlight exceptional cases
        m_flag = _get_convergence_flag(m_epochs, m_best, m_converged)
        d_flag = _get_convergence_flag(d_epochs, d_best, d_converged)
        
        print(f"{r.missing_fraction:<10.0%} {m_epochs:>10} {m_best:>10} {m_time:>10.1f} {m_flag:>6} "
              f"{d_epochs:>10} {d_best:>10} {d_time:>10.1f} {d_flag:>6}")
    
    print("-" * 90)
    print("Flags: ⚡=fast convergence (<100 epochs), 🎯=optimal (best near end), ⚠️=slow (>80% epochs), 🔴=no convergence")
    
    # Summary - neutral reporting
    modula_wins = len(results) - diffgeo_wins
    print(f"\nResults: DiffGeo {diffgeo_wins} wins, Modula {modula_wins} wins")
    print(f"Average RMSE difference: {avg_improvement:+.1f}% (positive = DiffGeo better)")
    
    if avg_improvement > 1:
        print("\n  DiffGeo had lower average RMSE in this comparison")
    elif avg_improvement < -1:
        print("\n  Modula had lower average RMSE in this comparison")
    else:
        print("\n  Methods performed similarly (within 1% RMSE)")

