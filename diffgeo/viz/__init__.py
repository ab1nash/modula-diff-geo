"""
Interactive Manifold Visualizer

Lightweight visualization tools for exploring learned geometric structures:
- SPD manifolds (covariance ellipsoids)
- Fisher Information (stiff/sloppy directions)
- Finsler/Randers metrics (asymmetric distances)
- Geodesics on manifolds

Uses Plotly for interactive 3D visualization in Jupyter notebooks.

Usage:
    from diffgeo.viz import ManifoldVisualizer, SPDViz, FinslerViz
    
    # Quick SPD visualization
    viz = SPDViz()
    viz.plot_ellipsoids(covariance_matrices)
    
    # Finsler metric with drift
    fviz = FinslerViz()
    fviz.plot_indicatrix(randers_metric)
"""

from .core import (
    ellipsoid_mesh,
    arrow_trace,
    colorscale_from_eigenvalues,
    setup_3d_layout,
)

from .spd_viz import SPDViz

from .finsler_viz import FinslerViz

from .interactive import ManifoldVisualizer

__all__ = [
    'ellipsoid_mesh',
    'arrow_trace',
    'colorscale_from_eigenvalues',
    'setup_3d_layout',
    'SPDViz',
    'FinslerViz',
    'ManifoldVisualizer',
]

