"""
SPD Manifold Visualization

Visualize Symmetric Positive Definite matrices as ellipsoids and explore
the Riemannian geometry of covariance matrices.

Features:
- Ellipsoid visualization of covariance matrices
- Fréchet mean vs arithmetic mean comparison
- Geodesic interpolation
- Tangent space projection visualization
"""
import numpy as np
from typing import List, Optional, Tuple, Union

from .core import (
    _check_plotly,
    ellipsoid_trace,
    arrow_trace,
    geodesic_path_trace,
    point_trace,
    setup_3d_layout,
    PLOTLY_AVAILABLE,
)

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


class SPDViz:
    """
    Visualizer for SPD (Symmetric Positive Definite) manifolds.
    
    SPD matrices appear in:
    - Covariance estimation (EEG, finance)
    - Diffusion tensors (brain imaging)
    - Gaussian distributions
    
    The key insight: SPD matrices form a curved space where:
    - Euclidean mean causes "swelling" (inflated determinant)
    - Riemannian mean preserves geometric properties
    
    Usage:
        viz = SPDViz()
        fig = viz.plot_ellipsoids([cov1, cov2, cov3])
        fig.show()
        
        # Compare means
        fig = viz.compare_means([cov1, cov2, cov3, cov4])
        fig.show()
    """
    
    def __init__(self, colorscale: str = 'Viridis'):
        """
        Args:
            colorscale: Default Plotly colorscale for visualizations
        """
        _check_plotly()
        self.colorscale = colorscale
        self._colors = [
            'rgba(99, 110, 250, 0.6)',   # Blue
            'rgba(239, 85, 59, 0.6)',    # Red
            'rgba(0, 204, 150, 0.6)',    # Green
            'rgba(171, 99, 250, 0.6)',   # Purple
            'rgba(255, 161, 90, 0.6)',   # Orange
            'rgba(25, 211, 243, 0.6)',   # Cyan
        ]
    
    def _get_color(self, idx: int) -> str:
        """Get color by index with cycling."""
        return self._colors[idx % len(self._colors)]
    
    def plot_ellipsoids(
        self,
        matrices: List[np.ndarray],
        centers: List[np.ndarray] = None,
        names: List[str] = None,
        scale: float = 1.0,
        title: str = 'SPD Matrices as Ellipsoids'
    ) -> 'go.Figure':
        """
        Plot multiple SPD matrices as ellipsoids.
        
        Each ellipsoid represents the set {x : x^T A^{-1} x = 1},
        which is the unit ball in the metric defined by A.
        
        Args:
            matrices: List of (3, 3) SPD matrices
            centers: Optional list of center points (default: spaced apart)
            names: Optional names for legend
            scale: Scale factor for ellipsoid size
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        n = len(matrices)
        
        # Default centers: spread along x-axis
        if centers is None:
            spacing = 3.0 * scale
            centers = [np.array([i * spacing, 0, 0]) for i in range(n)]
        
        # Default names
        if names is None:
            names = [f'Matrix {i+1}' for i in range(n)]
        
        traces = []
        for i, (mat, center, name) in enumerate(zip(matrices, centers, names)):
            trace = ellipsoid_trace(
                mat, center,
                color=self._get_color(i),
                name=name,
                scale=scale
            )
            traces.append(trace)
            
            # Add principal axes
            eigenvalues, eigenvectors = np.linalg.eigh(mat)
            for j in range(3):
                direction = eigenvectors[:, j] * np.sqrt(eigenvalues[j]) * scale
                arrows = arrow_trace(
                    center, direction,
                    color=self._get_color(i).replace('0.6', '1.0'),
                    name=f'{name} axis {j+1}' if i == 0 and j == 0 else '',
                    width=3
                )
                # Only show in legend for first ellipsoid, first axis
                arrows[0].showlegend = (i == 0 and j == 0)
                traces.extend(arrows)
        
        fig = go.Figure(data=traces)
        
        # Compute range for equal aspect
        all_centers = np.array(centers)
        max_range = max(3.0 * scale, np.ptp(all_centers[:, 0]) + 4 * scale)
        
        layout = setup_3d_layout(title)
        layout['scene']['aspectmode'] = 'cube'
        fig.update_layout(**layout)
        
        return fig
    
    def compare_means(
        self,
        matrices: List[np.ndarray],
        title: str = 'Euclidean vs Riemannian Mean'
    ) -> 'go.Figure':
        """
        Visualize the difference between Euclidean and Riemannian (Fréchet) means.
        
        The Euclidean mean of SPD matrices suffers from "swelling" - 
        the determinant increases beyond what geometry predicts.
        The Riemannian mean preserves geometric properties.
        
        Args:
            matrices: List of (3, 3) SPD matrices
            title: Plot title
            
        Returns:
            Plotly Figure with side-by-side comparison
        """
        # Compute means
        euclidean_mean = np.mean(matrices, axis=0)
        riemannian_mean = self._frechet_mean(matrices)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=['Euclidean Mean (swelling!)', 'Riemannian Mean (correct)']
        )
        
        # Left plot: Original + Euclidean mean
        center = np.zeros(3)
        for i, mat in enumerate(matrices):
            trace = ellipsoid_trace(
                mat, center,
                color=self._get_color(i),
                name=f'Matrix {i+1}',
                scale=0.8
            )
            fig.add_trace(trace, row=1, col=1)
        
        # Euclidean mean
        euc_trace = ellipsoid_trace(
            euclidean_mean, center,
            color='rgba(255, 0, 0, 0.4)',
            name='Euclidean Mean',
            scale=0.8
        )
        fig.add_trace(euc_trace, row=1, col=1)
        
        # Right plot: Original + Riemannian mean
        for i, mat in enumerate(matrices):
            trace = ellipsoid_trace(
                mat, center,
                color=self._get_color(i),
                name=f'Matrix {i+1}',
                showlegend=False,
                scale=0.8
            )
            fig.add_trace(trace, row=1, col=2)
        
        # Riemannian mean
        riem_trace = ellipsoid_trace(
            riemannian_mean, center,
            color='rgba(0, 255, 0, 0.4)',
            name='Riemannian Mean',
            scale=0.8
        )
        fig.add_trace(riem_trace, row=1, col=2)
        
        # Layout
        scene_config = dict(
            aspectmode='cube',
            xaxis=dict(range=[-2, 2]),
            yaxis=dict(range=[-2, 2]),
            zaxis=dict(range=[-2, 2])
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=scene_config,
            scene2=scene_config,
            showlegend=True,
            height=500,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        # Add determinant annotation
        det_euc = np.linalg.det(euclidean_mean)
        det_riem = np.linalg.det(riemannian_mean)
        det_orig = np.mean([np.linalg.det(m) for m in matrices])
        
        fig.add_annotation(
            text=f'Mean det: {det_euc:.3f} (inputs avg: {det_orig:.3f})',
            xref='paper', yref='paper',
            x=0.25, y=-0.05, showarrow=False
        )
        fig.add_annotation(
            text=f'Mean det: {det_riem:.3f} (inputs avg: {det_orig:.3f})',
            xref='paper', yref='paper',
            x=0.75, y=-0.05, showarrow=False
        )
        
        return fig
    
    def plot_geodesic(
        self,
        A: np.ndarray,
        B: np.ndarray,
        n_steps: int = 10,
        title: str = 'Geodesic on SPD Manifold'
    ) -> 'go.Figure':
        """
        Visualize geodesic interpolation between two SPD matrices.
        
        The geodesic γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}
        gives the shortest path on the SPD manifold.
        
        Args:
            A: Starting (3, 3) SPD matrix
            B: Ending (3, 3) SPD matrix
            n_steps: Number of interpolation steps
            title: Plot title
            
        Returns:
            Plotly Figure showing geodesic as sequence of ellipsoids
        """
        traces = []
        
        # Compute geodesic points
        t_values = np.linspace(0, 1, n_steps)
        geodesic_matrices = [self._geodesic(A, B, t) for t in t_values]
        
        # Place ellipsoids along a path
        for i, (t, mat) in enumerate(zip(t_values, geodesic_matrices)):
            center = np.array([t * 4, 0, 0])  # Spread along x
            opacity = 0.3 + 0.5 * (1 - abs(t - 0.5) * 2)  # Brighter at ends
            
            color = f'rgba({int(255*(1-t))}, {int(100 + 155*t)}, {int(255*t)}, {opacity})'
            
            trace = ellipsoid_trace(
                mat, center,
                color=color,
                name=f't={t:.2f}',
                scale=0.6,
                showlegend=(i % 3 == 0)  # Show every 3rd
            )
            traces.append(trace)
        
        fig = go.Figure(data=traces)
        
        layout = setup_3d_layout(title)
        layout['scene']['camera'] = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        fig.update_layout(**layout)
        
        return fig
    
    def plot_tangent_space(
        self,
        matrices: List[np.ndarray],
        labels: np.ndarray = None,
        title: str = 'Tangent Space Projection'
    ) -> 'go.Figure':
        """
        Project SPD matrices to tangent space and visualize.
        
        This is the key operation in Riemannian classification:
        1. Compute Fréchet mean of data
        2. Project all matrices to tangent space at mean
        3. Visualize in the flattened space
        
        Args:
            matrices: List of SPD matrices
            labels: Optional class labels for coloring
            title: Plot title
            
        Returns:
            Plotly Figure showing tangent space embedding
        """
        # Compute mean
        mean = self._frechet_mean(matrices)
        
        # Project to tangent space
        tangent_vectors = []
        for mat in matrices:
            V = self._log_map(mean, mat)
            # Extract upper triangle (symmetric, so redundant)
            idx = np.triu_indices(3)
            tangent_vectors.append(V[idx])
        
        tangent_vectors = np.array(tangent_vectors)
        
        # PCA to 3D for visualization
        centered = tangent_vectors - tangent_vectors.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:3].T
        
        # Create scatter plot
        if labels is not None:
            colors = labels
        else:
            colors = np.arange(len(matrices))
        
        trace = point_trace(
            projected, colors=colors,
            size=10, name='Tangent Vectors',
            colorscale=self.colorscale
        )
        
        fig = go.Figure(data=[trace])
        
        layout = setup_3d_layout(title)
        layout['scene']['xaxis']['title'] = 'PC1'
        layout['scene']['yaxis']['title'] = 'PC2'
        layout['scene']['zaxis']['title'] = 'PC3'
        fig.update_layout(**layout)
        
        return fig
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal SPD geometry computations
    # ─────────────────────────────────────────────────────────────────────────
    
    def _matrix_sqrt(self, A: np.ndarray) -> np.ndarray:
        """Matrix square root via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    def _matrix_sqrt_inv(self, A: np.ndarray) -> np.ndarray:
        """Inverse matrix square root."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    def _matrix_log(self, A: np.ndarray) -> np.ndarray:
        """Matrix logarithm for SPD matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    
    def _matrix_exp(self, A: np.ndarray) -> np.ndarray:
        """Matrix exponential for symmetric matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
    
    def _matrix_power(self, A: np.ndarray, t: float) -> np.ndarray:
        """Compute A^t for SPD matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(np.power(eigvals, t)) @ eigvecs.T
    
    def _geodesic(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Geodesic interpolation: γ(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}"""
        A_sqrt = self._matrix_sqrt(A)
        A_inv_sqrt = self._matrix_sqrt_inv(A)
        
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        inner_power_t = self._matrix_power(inner, t)
        
        return A_sqrt @ inner_power_t @ A_sqrt
    
    def _log_map(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Log map: Project Q to tangent space at P."""
        P_sqrt = self._matrix_sqrt(P)
        P_inv_sqrt = self._matrix_sqrt_inv(P)
        
        inner = P_inv_sqrt @ Q @ P_inv_sqrt
        log_inner = self._matrix_log(inner)
        
        return P_sqrt @ log_inner @ P_sqrt
    
    def _frechet_mean(
        self,
        matrices: List[np.ndarray],
        max_iter: int = 50,
        tol: float = 1e-6
    ) -> np.ndarray:
        """Compute Fréchet (Riemannian) mean iteratively."""
        # Initialize with arithmetic mean
        M = np.mean(matrices, axis=0)
        
        for _ in range(max_iter):
            # Average in tangent space
            tangent_sum = np.zeros_like(M)
            for mat in matrices:
                tangent_sum += self._log_map(M, mat)
            tangent_sum /= len(matrices)
            
            # Check convergence
            if np.linalg.norm(tangent_sum) < tol:
                break
            
            # Update via exponential map
            M = self._exp_map(M, tangent_sum)
        
        return M
    
    def _exp_map(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Exponential map: Move from P in direction V."""
        P_sqrt = self._matrix_sqrt(P)
        P_inv_sqrt = self._matrix_sqrt_inv(P)
        
        inner = P_inv_sqrt @ V @ P_inv_sqrt
        exp_inner = self._matrix_exp(inner)
        
        return P_sqrt @ exp_inner @ P_sqrt


__all__ = ['SPDViz']

