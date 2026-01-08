"""
Finsler Metric Visualization

Visualize asymmetric Finsler metrics, particularly Randers metrics
which combine Riemannian geometry with a drift vector.

Features:
- Indicatrix (unit ball) visualization showing asymmetry
- Drift vector overlay
- Forward vs backward distance comparison
- Geodesic asymmetry demonstration
"""
import numpy as np
from typing import List, Optional, Tuple, Union

from .core import (
    _check_plotly,
    arrow_trace,
    setup_3d_layout,
    PLOTLY_AVAILABLE,
)

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


class FinslerViz:
    """
    Visualizer for Finsler metrics, especially Randers metrics.
    
    Finsler metrics generalize Riemannian metrics to allow asymmetry:
    F(v) ≠ F(-v) in general.
    
    The Randers metric F(v) = √(v^T A v) + b^T v combines:
    - A: Symmetric positive-definite (Riemannian part)
    - b: Drift vector ("wind" that makes one direction cheaper)
    
    Physical interpretation:
    - Moving with the drift is cheaper
    - Moving against the drift is more expensive
    - Applications: directed graphs, causal modeling, thermodynamics
    
    Usage:
        from diffgeo.geometry import RandersMetric
        
        metric = RandersMetric(A, b)
        viz = FinslerViz()
        fig = viz.plot_indicatrix(metric)
        fig.show()
    """
    
    def __init__(self):
        _check_plotly()
    
    def plot_indicatrix(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_points: int = 50,
        title: str = 'Randers Indicatrix (Unit Ball)',
        show_riemannian: bool = True,
        show_drift: bool = True
    ) -> 'go.Figure':
        """
        Plot the indicatrix (unit ball) of a Randers metric.
        
        The indicatrix is {v : F(v) = 1}, which for asymmetric metrics
        is NOT centered at the origin.
        
        Args:
            A: (3, 3) SPD matrix (Riemannian part)
            b: (3,) drift vector (must satisfy |b|_A < 1)
            n_points: Resolution of the surface
            title: Plot title
            show_riemannian: Show the symmetric (Riemannian) indicatrix for comparison
            show_drift: Show the drift vector as an arrow
            
        Returns:
            Plotly Figure
        """
        traces = []
        
        # Compute Randers indicatrix
        X, Y, Z = self._compute_randers_indicatrix(A, b, n_points)
        
        randers_trace = go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'rgba(255, 100, 100, 0.7)'], [1, 'rgba(100, 100, 255, 0.7)']],
            surfacecolor=Z,  # Color by z to show asymmetry
            showscale=False,
            name='Randers Indicatrix',
            opacity=0.7,
            hoverinfo='name'
        )
        traces.append(randers_trace)
        
        # Riemannian comparison (b=0)
        if show_riemannian:
            X_r, Y_r, Z_r = self._compute_randers_indicatrix(A, np.zeros(3), n_points)
            riem_trace = go.Surface(
                x=X_r, y=Y_r, z=Z_r,
                colorscale=[[0, 'rgba(150, 150, 150, 0.3)'], [1, 'rgba(150, 150, 150, 0.3)']],
                showscale=False,
                name='Riemannian (b=0)',
                opacity=0.3,
                hoverinfo='name'
            )
            traces.append(riem_trace)
        
        # Drift vector
        if show_drift and np.linalg.norm(b) > 1e-8:
            origin = np.zeros(3)
            # Scale drift for visualization
            drift_scaled = b * 2 / (np.linalg.norm(b) + 1e-8)
            drift_arrows = arrow_trace(
                origin, drift_scaled,
                color='red',
                name=f'Drift b (|b|={np.linalg.norm(b):.2f})',
                width=5
            )
            traces.extend(drift_arrows)
        
        fig = go.Figure(data=traces)
        
        layout = setup_3d_layout(title)
        layout['scene']['aspectmode'] = 'cube'
        
        # Center view on indicatrix
        max_val = max(np.abs(X).max(), np.abs(Y).max(), np.abs(Z).max(), 2.0)
        for axis in ['xaxis', 'yaxis', 'zaxis']:
            layout['scene'][axis]['range'] = [-max_val, max_val]
        
        fig.update_layout(**layout)
        
        return fig
    
    def plot_asymmetry_comparison(
        self,
        A: np.ndarray,
        b: np.ndarray,
        title: str = 'Forward vs Backward Distances'
    ) -> 'go.Figure':
        """
        Visualize the asymmetry of a Randers metric.
        
        Shows how F(v) ≠ F(-v) by plotting distances in opposite directions.
        
        Args:
            A: (3, 3) SPD matrix
            b: (3,) drift vector
            title: Plot title
            
        Returns:
            Plotly Figure with asymmetry visualization
        """
        # Sample directions on unit sphere
        n_samples = 200
        phi = np.random.uniform(0, 2*np.pi, n_samples)
        theta = np.random.uniform(0, np.pi, n_samples)
        
        directions = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)
        
        # Compute forward and backward norms
        forward_norms = []
        backward_norms = []
        
        for v in directions:
            forward_norms.append(self._randers_norm(A, b, v))
            backward_norms.append(self._randers_norm(A, b, -v))
        
        forward_norms = np.array(forward_norms)
        backward_norms = np.array(backward_norms)
        
        # Asymmetry ratio
        asymmetry = forward_norms / (backward_norms + 1e-8)
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=['Directional Asymmetry', 'Asymmetry Distribution']
        )
        
        # 3D scatter colored by asymmetry
        scatter3d = go.Scatter3d(
            x=directions[:, 0],
            y=directions[:, 1],
            z=directions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=asymmetry,
                colorscale='RdBu',
                colorbar=dict(title='F(v)/F(-v)', x=0.45),
                cmin=0.5,
                cmax=1.5
            ),
            name='Asymmetry',
            hovertemplate='Asymmetry: %{marker.color:.2f}<extra></extra>'
        )
        fig.add_trace(scatter3d, row=1, col=1)
        
        # Histogram
        hist = go.Histogram(
            x=asymmetry,
            nbinsx=30,
            marker=dict(color='steelblue'),
            name='Asymmetry Ratio'
        )
        fig.add_trace(hist, row=1, col=2)
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=500,
            showlegend=False
        )
        
        # Add reference line at 1 (symmetric) - use add_shape directly for mixed subplot types
        fig.add_shape(
            type='line',
            x0=1.0, x1=1.0, y0=0, y1=1,
            xref='x2', yref='y2 domain',
            line=dict(color='red', dash='dash')
        )
        fig.add_annotation(
            text='Symmetric (ratio=1)',
            x=1.0, y=0.95, xref='x2', yref='paper',
            showarrow=False, font=dict(size=10)
        )
        
        return fig
    
    def plot_geodesic_comparison(
        self,
        A: np.ndarray,
        b: np.ndarray,
        p1: np.ndarray = None,
        p2: np.ndarray = None,
        title: str = 'Finsler Geodesic Asymmetry'
    ) -> 'go.Figure':
        """
        Compare geodesics in forward vs backward directions.
        
        For Finsler metrics, the "shortest path" from A to B may differ
        from B to A, reflecting the asymmetric cost structure.
        
        Args:
            A: (3, 3) SPD matrix
            b: (3,) drift vector  
            p1: Start point (default: origin)
            p2: End point (default: [2, 1, 0])
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        if p1 is None:
            p1 = np.zeros(3)
        if p2 is None:
            p2 = np.array([2.0, 1.0, 0.5])
        
        traces = []
        
        # Forward path (approximate geodesic)
        n_steps = 20
        forward_path = self._approximate_geodesic(A, b, p1, p2, n_steps)
        forward_trace = go.Scatter3d(
            x=forward_path[:, 0],
            y=forward_path[:, 1],
            z=forward_path[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=3, color='blue'),
            name=f'Forward (cost={self._path_length(A, b, forward_path):.2f})'
        )
        traces.append(forward_trace)
        
        # Backward path
        backward_path = self._approximate_geodesic(A, b, p2, p1, n_steps)
        backward_trace = go.Scatter3d(
            x=backward_path[:, 0],
            y=backward_path[:, 1],
            z=backward_path[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=3, color='red'),
            name=f'Backward (cost={self._path_length(A, b, backward_path):.2f})'
        )
        traces.append(backward_trace)
        
        # Endpoints
        endpoints = go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='markers+text',
            marker=dict(size=10, color=['green', 'orange']),
            text=['Start', 'End'],
            textposition='top center',
            name='Endpoints'
        )
        traces.append(endpoints)
        
        # Drift vector at origin
        if np.linalg.norm(b) > 1e-8:
            drift_arrows = arrow_trace(
                np.zeros(3), b * 2,
                color='purple',
                name='Drift',
                width=4
            )
            traces.extend(drift_arrows)
        
        fig = go.Figure(data=traces)
        
        layout = setup_3d_layout(title)
        fig.update_layout(**layout)
        
        return fig
    
    def plot_fisher_metric(
        self,
        fisher_matrix: np.ndarray,
        eigenvalue_threshold: float = 0.01,
        title: str = 'Fisher Information Geometry'
    ) -> 'go.Figure':
        """
        Visualize Fisher Information matrix showing stiff vs sloppy directions.
        
        From the research documents:
        - "Stiff" directions: Large eigenvalues, well-constrained by data
        - "Sloppy" directions: Small eigenvalues, parameters can vary freely
        
        Args:
            fisher_matrix: (n, n) Fisher Information Matrix
            eigenvalue_threshold: Threshold for "sloppy" classification
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(fisher_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Classify directions
        n_stiff = np.sum(eigenvalues > eigenvalue_threshold * eigenvalues.max())
        n_sloppy = len(eigenvalues) - n_stiff
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=['Principal Directions', 'Eigenvalue Spectrum']
        )
        
        # 3D arrows for principal directions (top 3 or fewer)
        n_show = min(3, len(eigenvalues))
        origin = np.zeros(3)
        
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
        
        for i in range(n_show):
            direction = eigenvectors[:n_show, i] * np.sqrt(eigenvalues[i])
            direction_3d = np.zeros(3)
            direction_3d[:len(direction)] = direction[:3]
            
            arrows = arrow_trace(
                origin, direction_3d,
                color=colors[i % len(colors)],
                name=f'λ_{i+1}={eigenvalues[i]:.2e}',
                width=4
            )
            for arrow in arrows:
                fig.add_trace(arrow, row=1, col=1)
        
        # Eigenvalue spectrum
        bar = go.Bar(
            x=list(range(1, len(eigenvalues) + 1)),
            y=eigenvalues,
            marker=dict(
                color=['red' if e > eigenvalue_threshold * eigenvalues.max() else 'lightblue' 
                       for e in eigenvalues]
            ),
            name='Eigenvalues'
        )
        fig.add_trace(bar, row=1, col=2)
        
        fig.update_layout(
            title=dict(text=f'{title}<br><sub>{n_stiff} stiff, {n_sloppy} sloppy directions</sub>', x=0.5),
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(type='log', title='Eigenvalue', row=1, col=2)
        fig.update_xaxes(title='Index', row=1, col=2)
        
        return fig
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal computations
    # ─────────────────────────────────────────────────────────────────────────
    
    def _randers_norm(self, A: np.ndarray, b: np.ndarray, v: np.ndarray) -> float:
        """Compute Randers norm F(v) = sqrt(v^T A v) + b^T v"""
        riemannian_part = np.sqrt(max(v @ A @ v, 1e-10))
        drift_part = b @ v
        return riemannian_part + drift_part
    
    def _compute_randers_indicatrix(
        self,
        A: np.ndarray,
        b: np.ndarray,
        n_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the indicatrix surface {v : F(v) = 1}.
        
        For Randers metrics, we parameterize by direction and solve for radius.
        """
        # Parameterize directions
        u = np.linspace(0, 2 * np.pi, n_points)
        v_param = np.linspace(0.01, np.pi - 0.01, n_points)  # Avoid poles
        
        X = np.zeros((n_points, n_points))
        Y = np.zeros((n_points, n_points))
        Z = np.zeros((n_points, n_points))
        
        for i, phi in enumerate(u):
            for j, theta in enumerate(v_param):
                # Unit direction
                direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                # Solve F(r * direction) = 1 for r
                # sqrt(r^2 * d^T A d) + r * b^T d = 1
                # r * sqrt(d^T A d) + r * b^T d = 1
                # r * (sqrt(d^T A d) + b^T d) = 1
                
                dAd = direction @ A @ direction
                bd = b @ direction
                
                # r = 1 / (sqrt(dAd) + bd)
                denominator = np.sqrt(max(dAd, 1e-10)) + bd
                
                if denominator > 1e-8:
                    r = 1.0 / denominator
                else:
                    r = 10.0  # Large value for nearly zero denominator
                
                # Clamp for visualization
                r = min(r, 10.0)
                r = max(r, 0.01)
                
                X[i, j] = r * direction[0]
                Y[i, j] = r * direction[1]
                Z[i, j] = r * direction[2]
        
        return X, Y, Z
    
    def _approximate_geodesic(
        self,
        A: np.ndarray,
        b: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Approximate geodesic using linear interpolation with Finsler adjustment.
        
        This is a first-order approximation that accounts for drift.
        """
        displacement = p2 - p1
        
        # Compute asymmetry factor
        forward_norm = self._randers_norm(A, b, displacement / (np.linalg.norm(displacement) + 1e-8))
        backward_norm = self._randers_norm(A, b, -displacement / (np.linalg.norm(displacement) + 1e-8))
        
        asymmetry = forward_norm / (backward_norm + 1e-8)
        
        # Adjusted interpolation
        path = []
        for i in range(n_steps + 1):
            t = i / n_steps
            # Adjust t based on asymmetry
            t_adj = t ** (1.0 / asymmetry) if asymmetry > 0.1 else t
            path.append(p1 + t_adj * displacement)
        
        return np.array(path)
    
    def _path_length(self, A: np.ndarray, b: np.ndarray, path: np.ndarray) -> float:
        """Compute total Finsler length of a path."""
        length = 0.0
        for i in range(len(path) - 1):
            segment = path[i + 1] - path[i]
            length += self._randers_norm(A, b, segment)
        return length


__all__ = ['FinslerViz']

