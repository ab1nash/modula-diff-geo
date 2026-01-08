"""
Core visualization primitives for manifold visualization.

Provides low-level building blocks:
- Ellipsoid mesh generation from SPD matrices
- Arrow/vector visualization
- Color mapping utilities
- 3D layout configuration
"""
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

# Try to import plotly, provide helpful error if missing
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


def _check_plotly():
    """Raise helpful error if plotly not installed."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. Install with:\n"
            "  pip install plotly\n"
            "Or for Jupyter support:\n"
            "  pip install plotly nbformat"
        )


def ellipsoid_mesh(
    matrix: np.ndarray,
    center: np.ndarray = None,
    n_points: int = 20,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mesh coordinates for an ellipsoid defined by SPD matrix.
    
    The ellipsoid is the set {x : x^T A^{-1} x = 1}, which corresponds
    to the unit ball in the metric defined by A.
    
    Args:
        matrix: (n, n) SPD matrix defining the ellipsoid shape
        center: (n,) center point (default: origin)
        n_points: Resolution of the mesh
        scale: Scaling factor for visualization
        
    Returns:
        Tuple of (X, Y, Z) mesh coordinates
    """
    if center is None:
        center = np.zeros(matrix.shape[0])
    
    # Eigendecomposition: A = V @ D @ V.T
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
    
    # Semi-axes lengths are sqrt of eigenvalues
    axes = np.sqrt(eigenvalues) * scale
    
    # Create unit sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Stack and transform
    sphere = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
    
    # Scale by axes
    scaled = np.diag(axes) @ sphere
    
    # Rotate by eigenvectors
    rotated = eigenvectors @ scaled
    
    # Translate to center
    X = rotated[0].reshape(n_points, n_points) + center[0]
    Y = rotated[1].reshape(n_points, n_points) + center[1]
    Z = rotated[2].reshape(n_points, n_points) + center[2]
    
    return X, Y, Z


def ellipsoid_trace(
    matrix: np.ndarray,
    center: np.ndarray = None,
    color: str = 'rgba(100, 150, 255, 0.5)',
    name: str = 'Ellipsoid',
    n_points: int = 20,
    scale: float = 1.0,
    showlegend: bool = True
) -> 'go.Surface':
    """
    Create Plotly Surface trace for an ellipsoid.
    
    Args:
        matrix: (3, 3) SPD matrix
        center: (3,) center point
        color: Surface color (supports rgba for transparency)
        name: Trace name for legend
        n_points: Mesh resolution
        scale: Size scaling
        showlegend: Whether to show in legend
        
    Returns:
        Plotly Surface trace
    """
    _check_plotly()
    
    X, Y, Z = ellipsoid_mesh(matrix, center, n_points, scale)
    
    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        showlegend=showlegend,
        opacity=0.6,
        hoverinfo='name'
    )


def arrow_trace(
    start: np.ndarray,
    direction: np.ndarray,
    color: str = 'red',
    name: str = 'Vector',
    width: float = 4,
    cone_ratio: float = 0.15
) -> List['go.Scatter3d']:
    """
    Create arrow (vector) visualization as line + cone.
    
    Args:
        start: (3,) starting point
        direction: (3,) direction vector
        color: Arrow color
        name: Trace name
        width: Line width
        cone_ratio: Ratio of cone size to arrow length
        
    Returns:
        List of [line trace, cone trace]
    """
    _check_plotly()
    
    end = start + direction
    
    # Line segment
    line = go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        showlegend=True,
        hoverinfo='name'
    )
    
    # Cone (arrowhead)
    cone = go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=np.linalg.norm(direction) * cone_ratio,
        anchor='tail',
        showlegend=False,
        hoverinfo='skip'
    )
    
    return [line, cone]


def colorscale_from_eigenvalues(
    eigenvalues: np.ndarray,
    cmap: str = 'Viridis'
) -> List[Tuple[float, str]]:
    """
    Create color scale based on eigenvalue magnitudes.
    
    Useful for coloring ellipsoids by their "stiffness" - 
    larger eigenvalues = stiffer directions = brighter colors.
    
    Args:
        eigenvalues: Array of eigenvalues
        cmap: Matplotlib colormap name
        
    Returns:
        Plotly-compatible colorscale
    """
    import matplotlib.pyplot as plt
    
    cmap_obj = plt.get_cmap(cmap)
    
    # Normalize eigenvalues
    eig_min, eig_max = eigenvalues.min(), eigenvalues.max()
    if eig_max - eig_min < 1e-10:
        normalized = np.ones_like(eigenvalues) * 0.5
    else:
        normalized = (eigenvalues - eig_min) / (eig_max - eig_min)
    
    # Create colorscale
    colors = []
    for i, val in enumerate(np.linspace(0, 1, len(eigenvalues))):
        rgba = cmap_obj(normalized[i])
        colors.append((val, f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'))
    
    return colors


def setup_3d_layout(
    title: str = 'Manifold Visualization',
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    z_range: Tuple[float, float] = None,
    equal_aspect: bool = True
) -> Dict[str, Any]:
    """
    Create standard 3D layout configuration.
    
    Args:
        title: Plot title
        x_range, y_range, z_range: Axis ranges
        equal_aspect: Whether to use equal aspect ratio
        
    Returns:
        Layout dictionary for Plotly
    """
    scene = dict(
        xaxis=dict(title='X', showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(title='Y', showgrid=True, gridwidth=1, gridcolor='lightgray'),
        zaxis=dict(title='Z', showgrid=True, gridwidth=1, gridcolor='lightgray'),
        bgcolor='rgba(250, 250, 250, 1)'
    )
    
    if x_range:
        scene['xaxis']['range'] = x_range
    if y_range:
        scene['yaxis']['range'] = y_range
    if z_range:
        scene['zaxis']['range'] = z_range
        
    if equal_aspect:
        scene['aspectmode'] = 'cube'
    
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        scene=scene,
        showlegend=True,
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white'
    )


def geodesic_path_trace(
    points: np.ndarray,
    color: str = 'blue',
    name: str = 'Geodesic',
    width: float = 3,
    dash: str = None
) -> 'go.Scatter3d':
    """
    Create trace for a geodesic path.
    
    Args:
        points: (n_points, 3) array of points along the path
        color: Line color
        name: Trace name
        width: Line width
        dash: Line dash style ('solid', 'dash', 'dot', 'dashdot')
        
    Returns:
        Scatter3d trace
    """
    _check_plotly()
    
    line_dict = dict(color=color, width=width)
    if dash:
        line_dict['dash'] = dash
    
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='lines',
        line=line_dict,
        name=name,
        showlegend=True
    )


def point_trace(
    points: np.ndarray,
    colors: np.ndarray = None,
    size: float = 8,
    name: str = 'Points',
    colorscale: str = 'Viridis',
    showscale: bool = True
) -> 'go.Scatter3d':
    """
    Create scatter plot of 3D points.
    
    Args:
        points: (n, 3) array of points
        colors: Optional (n,) array for coloring
        size: Marker size
        name: Trace name
        colorscale: Plotly colorscale name
        showscale: Whether to show colorbar
        
    Returns:
        Scatter3d trace
    """
    _check_plotly()
    
    marker = dict(size=size, opacity=0.8)
    
    if colors is not None:
        marker['color'] = colors
        marker['colorscale'] = colorscale
        marker['showscale'] = showscale
        marker['colorbar'] = dict(title='Value')
    
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=marker,
        name=name,
        showlegend=True
    )


__all__ = [
    'ellipsoid_mesh',
    'ellipsoid_trace',
    'arrow_trace',
    'colorscale_from_eigenvalues',
    'setup_3d_layout',
    'geodesic_path_trace',
    'point_trace',
    'PLOTLY_AVAILABLE',
]

