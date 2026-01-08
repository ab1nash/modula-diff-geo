"""
Interactive Manifold Visualizer

Main interface for exploring learned geometric structures interactively.
Combines SPD, Finsler, and Fisher visualizations with parameter controls.

Works in Jupyter notebooks with ipywidgets for interactivity.
"""
import numpy as np
from typing import Optional, Dict, Any, Callable, Union

from .core import _check_plotly, PLOTLY_AVAILABLE, setup_3d_layout
from .spd_viz import SPDViz
from .finsler_viz import FinslerViz

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# Try to import ipywidgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None


class ManifoldVisualizer:
    """
    Interactive visualizer for learned manifolds.
    
    This is the main entry point for exploring geometric structures.
    It provides:
    - Real-time parameter adjustment via sliders
    - Multiple visualization modes (SPD, Finsler, Fisher)
    - Side-by-side comparisons
    - Export to HTML/PNG
    
    Usage:
        # Basic usage
        viz = ManifoldVisualizer()
        viz.explore_spd(matrices)
        
        # With Randers metric from your code
        from diffgeo.geometry import RandersMetric
        metric = RandersMetric(A, b)
        viz.explore_finsler(metric.A, metric.b)
        
        # Interactive in Jupyter
        viz.interactive_randers()  # Creates sliders for drift
    """
    
    def __init__(self):
        _check_plotly()
        self.spd_viz = SPDViz()
        self.finsler_viz = FinslerViz()
        self._current_fig = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPD EXPLORATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def explore_spd(
        self,
        matrices: np.ndarray,
        mode: str = 'ellipsoids'
    ) -> 'go.Figure':
        """
        Explore SPD matrices with various visualization modes.
        
        Args:
            matrices: (n, 3, 3) array of SPD matrices
            mode: 'ellipsoids', 'geodesic', 'means', or 'tangent'
            
        Returns:
            Plotly Figure
        """
        matrices = [m for m in np.asarray(matrices)]
        
        if mode == 'ellipsoids':
            fig = self.spd_viz.plot_ellipsoids(matrices)
        elif mode == 'geodesic' and len(matrices) >= 2:
            fig = self.spd_viz.plot_geodesic(matrices[0], matrices[1])
        elif mode == 'means':
            fig = self.spd_viz.compare_means(matrices)
        elif mode == 'tangent':
            fig = self.spd_viz.plot_tangent_space(matrices)
        else:
            fig = self.spd_viz.plot_ellipsoids(matrices)
        
        self._current_fig = fig
        return fig
    
    def explore_finsler(
        self,
        A: np.ndarray,
        b: np.ndarray = None,
        mode: str = 'indicatrix'
    ) -> 'go.Figure':
        """
        Explore Finsler (Randers) metric geometry.
        
        Args:
            A: (3, 3) SPD matrix
            b: (3,) drift vector (default: zeros)
            mode: 'indicatrix', 'asymmetry', or 'geodesic'
            
        Returns:
            Plotly Figure
        """
        if b is None:
            b = np.zeros(3)
        
        if mode == 'indicatrix':
            fig = self.finsler_viz.plot_indicatrix(A, b)
        elif mode == 'asymmetry':
            fig = self.finsler_viz.plot_asymmetry_comparison(A, b)
        elif mode == 'geodesic':
            fig = self.finsler_viz.plot_geodesic_comparison(A, b)
        else:
            fig = self.finsler_viz.plot_indicatrix(A, b)
        
        self._current_fig = fig
        return fig
    
    def explore_fisher(
        self,
        fisher_matrix: np.ndarray,
        threshold: float = 0.01
    ) -> 'go.Figure':
        """
        Explore Fisher Information geometry.
        
        Args:
            fisher_matrix: (n, n) Fisher Information Matrix
            threshold: Eigenvalue threshold for stiff/sloppy classification
            
        Returns:
            Plotly Figure
        """
        fig = self.finsler_viz.plot_fisher_metric(fisher_matrix, threshold)
        self._current_fig = fig
        return fig
    
    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE WIDGETS
    # ─────────────────────────────────────────────────────────────────────────
    
    def interactive_randers(
        self,
        A: np.ndarray = None,
        initial_drift: float = 0.3
    ):
        """
        Interactive Randers metric explorer with drift sliders.
        
        Creates sliders to adjust:
        - Drift strength (0 to 0.9)
        - Drift direction (spherical coordinates)
        - Visualization mode
        
        Args:
            A: (3, 3) SPD matrix (default: identity)
            initial_drift: Initial drift strength
        """
        if not WIDGETS_AVAILABLE:
            print("ipywidgets not available. Install with: pip install ipywidgets")
            print("Falling back to static visualization...")
            if A is None:
                A = np.eye(3)
            b = np.array([initial_drift, 0, 0])
            return self.explore_finsler(A, b)
        
        if A is None:
            A = np.eye(3)
        
        # Create widgets
        drift_strength = widgets.FloatSlider(
            value=initial_drift,
            min=0.0,
            max=0.9,
            step=0.05,
            description='Drift |b|:',
            continuous_update=False
        )
        
        drift_theta = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=np.pi,
            step=0.1,
            description='θ (polar):',
            continuous_update=False
        )
        
        drift_phi = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=2*np.pi,
            step=0.1,
            description='φ (azimuth):',
            continuous_update=False
        )
        
        mode_dropdown = widgets.Dropdown(
            options=['indicatrix', 'asymmetry', 'geodesic'],
            value='indicatrix',
            description='Mode:'
        )
        
        output = widgets.Output()
        
        def update(change=None):
            # Compute drift vector from spherical coordinates
            strength = drift_strength.value
            theta = drift_theta.value
            phi = drift_phi.value
            
            b = strength * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            with output:
                clear_output(wait=True)
                fig = self.explore_finsler(A, b, mode=mode_dropdown.value)
                fig.show()
        
        # Connect callbacks
        drift_strength.observe(update, names='value')
        drift_theta.observe(update, names='value')
        drift_phi.observe(update, names='value')
        mode_dropdown.observe(update, names='value')
        
        # Layout
        controls = widgets.VBox([
            widgets.HBox([drift_strength, mode_dropdown]),
            widgets.HBox([drift_theta, drift_phi])
        ])
        
        display(controls, output)
        update()  # Initial render
    
    def interactive_spd(
        self,
        n_matrices: int = 3,
        scale_range: tuple = (0.5, 2.0)
    ):
        """
        Interactive SPD matrix explorer.
        
        Creates sliders to adjust eigenvalues of sample matrices
        and compare Euclidean vs Riemannian means.
        
        Args:
            n_matrices: Number of matrices to visualize
            scale_range: Range for eigenvalue scaling
        """
        if not WIDGETS_AVAILABLE:
            print("ipywidgets not available. Install with: pip install ipywidgets")
            print("Generating random SPD matrices...")
            matrices = self._random_spd_matrices(n_matrices)
            return self.explore_spd(matrices, mode='means')
        
        # Create eigenvalue sliders
        eigenvalue_sliders = []
        for i in range(n_matrices):
            slider = widgets.FloatRangeSlider(
                value=[0.5, 1.5],
                min=scale_range[0],
                max=scale_range[1],
                step=0.1,
                description=f'Matrix {i+1} λ:',
                continuous_update=False
            )
            eigenvalue_sliders.append(slider)
        
        mode_dropdown = widgets.Dropdown(
            options=['ellipsoids', 'means', 'geodesic', 'tangent'],
            value='means',
            description='Mode:'
        )
        
        output = widgets.Output()
        
        def update(change=None):
            # Generate matrices based on slider values
            matrices = []
            for i, slider in enumerate(eigenvalue_sliders):
                eig_min, eig_max = slider.value
                eigenvalues = np.random.uniform(eig_min, eig_max, 3)
                # Random rotation
                Q, _ = np.linalg.qr(np.random.randn(3, 3))
                mat = Q @ np.diag(eigenvalues) @ Q.T
                matrices.append(mat)
            
            with output:
                clear_output(wait=True)
                fig = self.explore_spd(np.array(matrices), mode=mode_dropdown.value)
                fig.show()
        
        # Connect callbacks
        for slider in eigenvalue_sliders:
            slider.observe(update, names='value')
        mode_dropdown.observe(update, names='value')
        
        # Layout
        controls = widgets.VBox([mode_dropdown] + eigenvalue_sliders)
        
        display(controls, output)
        update()
    
    def interactive_fisher(
        self,
        dim: int = 5,
        condition_range: tuple = (1, 1000)
    ):
        """
        Interactive Fisher metric explorer.
        
        Creates controls to adjust condition number and visualize
        stiff vs sloppy directions.
        
        Args:
            dim: Dimension of Fisher matrix
            condition_range: Range for condition number
        """
        if not WIDGETS_AVAILABLE:
            print("ipywidgets not available. Install with: pip install ipywidgets")
            fisher = self._random_fisher(dim, 100)
            return self.explore_fisher(fisher)
        
        condition_slider = widgets.FloatLogSlider(
            value=100,
            base=10,
            min=np.log10(condition_range[0]),
            max=np.log10(condition_range[1]),
            step=0.1,
            description='Cond #:',
            continuous_update=False
        )
        
        threshold_slider = widgets.FloatLogSlider(
            value=0.01,
            base=10,
            min=-4,
            max=0,
            step=0.1,
            description='Threshold:',
            continuous_update=False
        )
        
        output = widgets.Output()
        
        def update(change=None):
            # Generate Fisher matrix with specified condition number
            fisher = self._random_fisher(dim, condition_slider.value)
            
            with output:
                clear_output(wait=True)
                fig = self.explore_fisher(fisher, threshold=threshold_slider.value)
                fig.show()
        
        condition_slider.observe(update, names='value')
        threshold_slider.observe(update, names='value')
        
        controls = widgets.HBox([condition_slider, threshold_slider])
        
        display(controls, output)
        update()
    
    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _random_spd_matrices(self, n: int, dim: int = 3) -> np.ndarray:
        """Generate random SPD matrices."""
        matrices = []
        for _ in range(n):
            L = np.random.randn(dim, dim)
            mat = L @ L.T + 0.1 * np.eye(dim)
            matrices.append(mat)
        return np.array(matrices)
    
    def _random_fisher(self, dim: int, condition_number: float) -> np.ndarray:
        """Generate random Fisher matrix with specified condition number."""
        # Generate eigenvalues with log-uniform spacing
        log_min = 0
        log_max = np.log10(condition_number)
        eigenvalues = 10 ** np.linspace(log_min, log_max, dim)
        
        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        
        return Q @ np.diag(eigenvalues) @ Q.T
    
    def save_html(self, filename: str):
        """Save current figure to interactive HTML file."""
        if self._current_fig is None:
            raise ValueError("No figure to save. Run a visualization first.")
        self._current_fig.write_html(filename)
        print(f"Saved to {filename}")
    
    def save_png(self, filename: str, width: int = 1200, height: int = 800):
        """Save current figure to PNG image."""
        if self._current_fig is None:
            raise ValueError("No figure to save. Run a visualization first.")
        try:
            self._current_fig.write_image(filename, width=width, height=height)
            print(f"Saved to {filename}")
        except ValueError as e:
            print(f"Error saving PNG: {e}")
            print("You may need to install kaleido: pip install kaleido")


def quick_spd_demo():
    """Quick demonstration of SPD visualization."""
    viz = ManifoldVisualizer()
    
    # Generate sample covariance matrices
    matrices = []
    for _ in range(4):
        L = np.random.randn(3, 3)
        mat = L @ L.T + 0.5 * np.eye(3)
        matrices.append(mat)
    
    print("SPD Manifold Visualization Demo")
    print("=" * 40)
    
    fig = viz.explore_spd(matrices, mode='means')
    fig.show()
    
    return viz


def quick_finsler_demo():
    """Quick demonstration of Finsler visualization."""
    viz = ManifoldVisualizer()
    
    # Create Randers metric
    A = np.array([
        [2.0, 0.5, 0.0],
        [0.5, 1.5, 0.3],
        [0.0, 0.3, 1.0]
    ])
    b = np.array([0.3, 0.2, 0.0])
    
    print("Finsler (Randers) Metric Visualization Demo")
    print("=" * 40)
    
    fig = viz.explore_finsler(A, b, mode='indicatrix')
    fig.show()
    
    return viz


__all__ = [
    'ManifoldVisualizer',
    'quick_spd_demo',
    'quick_finsler_demo',
]

