"""
Manifold Integrity Score (MIS) - A Novel Metric for Geometric Data Quality

This module implements the Manifold Integrity Score, a continuous metric that
quantifies how well data respects manifold constraints after imputation.

Unlike binary validity checks, MIS provides:
1. Continuous measurement (not just pass/fail)
2. Interpretable scale (0 = perfect, higher = worse)
3. Comparable across different manifold types (normalized)
4. Decomposable into specific constraint violations

Reference: See docs/manifold_integrity_score.md for full documentation.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ManifoldType(Enum):
    """Supported manifold types."""
    SPD = "spd"              # Symmetric Positive Definite matrices
    SPHERE = "sphere"        # Unit sphere S^n
    SO3 = "so3"              # Special Orthogonal group (rotations)
    SE3 = "se3"              # Special Euclidean group (rigid transforms)
    STIEFEL = "stiefel"      # Stiefel manifold (orthonormal frames)


@dataclass
class ManifoldIntegrityResult:
    """
    Results from Manifold Integrity Score computation.
    
    Attributes:
        mis: Overall Manifold Integrity Score (0 = perfect)
        symmetry_violation: For SPD/symmetric matrices
        positivity_violation: For SPD (negative eigenvalues)
        orthogonality_violation: For SO(3), Stiefel
        determinant_violation: For SO(3) (det should be 1)
        norm_violation: For spherical data (||x|| should be 1)
        is_valid: Binary validity (MIS < threshold)
        details: Additional diagnostic information
    """
    mis: float
    symmetry_violation: float = 0.0
    positivity_violation: float = 0.0
    orthogonality_violation: float = 0.0
    determinant_violation: float = 0.0
    norm_violation: float = 0.0
    is_valid: bool = True
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ManifoldIntegrityScore:
    """
    Compute Manifold Integrity Score (MIS) for geometric data.
    
    MIS measures how well data respects manifold constraints:
    - MIS = 0: Data lies exactly on the manifold
    - MIS > 0: Data violates manifold constraints (larger = worse)
    
    The score is designed to be:
    1. Continuous and differentiable (useful for optimization)
    2. Interpretable (each component has geometric meaning)
    3. Normalized (comparable across manifold types)
    
    Example:
        >>> mis = ManifoldIntegrityScore(ManifoldType.SPD, dim=3)
        >>> result = mis.compute(matrix)
        >>> print(f"MIS: {result.mis:.4f}")
    """
    
    # Threshold for "valid" classification
    VALIDITY_THRESHOLD = 0.01
    
    def __init__(self, manifold_type: ManifoldType, dim: int = None):
        """
        Initialize MIS calculator.
        
        Args:
            manifold_type: Type of manifold
            dim: Dimension (matrix size for SPD, vector dim for sphere)
        """
        self.manifold_type = manifold_type
        self.dim = dim
    
    def compute(self, data: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for given data.
        
        Args:
            data: Data array (shape depends on manifold type)
            
        Returns:
            ManifoldIntegrityResult with scores and diagnostics
        """
        if self.manifold_type == ManifoldType.SPD:
            return self._compute_spd(data)
        elif self.manifold_type == ManifoldType.SPHERE:
            return self._compute_sphere(data)
        elif self.manifold_type == ManifoldType.SO3:
            return self._compute_so3(data)
        elif self.manifold_type == ManifoldType.SE3:
            return self._compute_se3(data)
        elif self.manifold_type == ManifoldType.STIEFEL:
            return self._compute_stiefel(data)
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
    
    def compute_batch(self, data: np.ndarray) -> Tuple[float, ManifoldIntegrityResult]:
        """
        Compute average MIS for a batch of data.
        
        Returns:
            (mean_mis, aggregated_result)
        """
        if data.ndim == 2 and self.manifold_type == ManifoldType.SPHERE:
            # Batch of vectors
            results = [self.compute(data[i]) for i in range(len(data))]
        elif data.ndim == 3:
            # Batch of matrices
            results = [self.compute(data[i]) for i in range(len(data))]
        else:
            results = [self.compute(data)]
        
        mean_mis = np.mean([r.mis for r in results])
        
        # Aggregate
        agg = ManifoldIntegrityResult(
            mis=mean_mis,
            symmetry_violation=np.mean([r.symmetry_violation for r in results]),
            positivity_violation=np.mean([r.positivity_violation for r in results]),
            orthogonality_violation=np.mean([r.orthogonality_violation for r in results]),
            determinant_violation=np.mean([r.determinant_violation for r in results]),
            norm_violation=np.mean([r.norm_violation for r in results]),
            is_valid=mean_mis < self.VALIDITY_THRESHOLD,
            details={'n_samples': len(results), 'validity_rate': np.mean([r.is_valid for r in results])}
        )
        
        return mean_mis, agg
    
    def _compute_spd(self, A: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for SPD matrix.
        
        SPD constraints:
        1. Symmetry: A = A^T
        2. Positive definiteness: all eigenvalues > 0
        
        MIS = w1 * symmetry_violation + w2 * positivity_violation
        """
        n = A.shape[0]
        
        # 1. Symmetry violation: ||A - A^T||_F / ||A||_F
        sym_diff = A - A.T
        sym_violation = np.linalg.norm(sym_diff, 'fro') / (np.linalg.norm(A, 'fro') + 1e-10)
        
        # 2. Positivity violation: sum of |min(λ_i, 0)| / n
        # Symmetrize first for eigenvalue computation
        A_sym = (A + A.T) / 2
        eigvals = np.linalg.eigvalsh(A_sym)
        
        # Measure how negative the eigenvalues are (normalized by dimension)
        negative_eigvals = np.minimum(eigvals, 0)
        pos_violation = np.sum(np.abs(negative_eigvals)) / (n * np.abs(eigvals).max() + 1e-10)
        
        # Also track condition number for numerical stability
        cond_number = np.abs(eigvals.max()) / (np.abs(eigvals.min()) + 1e-10)
        
        # Combined MIS (weighted sum)
        # Symmetry is critical, positivity is critical
        mis = 0.3 * sym_violation + 0.7 * pos_violation
        
        return ManifoldIntegrityResult(
            mis=float(mis),
            symmetry_violation=float(sym_violation),
            positivity_violation=float(pos_violation),
            is_valid=mis < self.VALIDITY_THRESHOLD,
            details={
                'min_eigenvalue': float(eigvals.min()),
                'max_eigenvalue': float(eigvals.max()),
                'condition_number': float(cond_number),
                'n_negative_eigenvalues': int(np.sum(eigvals < 0)),
            }
        )
    
    def _compute_sphere(self, x: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for spherical data (unit vectors).
        
        Sphere constraint: ||x|| = 1
        
        MIS = | ||x|| - 1 |
        """
        norm = np.linalg.norm(x)
        norm_violation = np.abs(norm - 1.0)
        
        mis = norm_violation
        
        return ManifoldIntegrityResult(
            mis=float(mis),
            norm_violation=float(norm_violation),
            is_valid=mis < self.VALIDITY_THRESHOLD,
            details={
                'actual_norm': float(norm),
                'target_norm': 1.0,
            }
        )
    
    def _compute_so3(self, R: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for rotation matrix (SO(3)).
        
        SO(3) constraints:
        1. Orthogonality: R^T R = I
        2. Unit determinant: det(R) = 1
        
        MIS = w1 * orthogonality_violation + w2 * determinant_violation
        """
        n = R.shape[0]
        
        # 1. Orthogonality violation: ||R^T R - I||_F / sqrt(n)
        RtR = R.T @ R
        orth_diff = RtR - np.eye(n)
        orth_violation = np.linalg.norm(orth_diff, 'fro') / np.sqrt(n)
        
        # 2. Determinant violation: |det(R) - 1|
        det = np.linalg.det(R)
        det_violation = np.abs(det - 1.0)
        
        # Combined MIS
        mis = 0.7 * orth_violation + 0.3 * det_violation
        
        return ManifoldIntegrityResult(
            mis=float(mis),
            orthogonality_violation=float(orth_violation),
            determinant_violation=float(det_violation),
            is_valid=mis < self.VALIDITY_THRESHOLD,
            details={
                'determinant': float(det),
                'max_orth_deviation': float(np.abs(orth_diff).max()),
            }
        )
    
    def _compute_se3(self, T: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for rigid transformation (SE(3)).
        
        SE(3) = SO(3) × R³, so we check the rotation part.
        T is 4x4 homogeneous transformation matrix.
        """
        # Extract rotation part
        R = T[:3, :3]
        
        # Check rotation constraints
        so3_result = self._compute_so3(R)
        
        # Check bottom row [0, 0, 0, 1]
        bottom_row = T[3, :]
        bottom_violation = np.linalg.norm(bottom_row - np.array([0, 0, 0, 1]))
        
        mis = 0.8 * so3_result.mis + 0.2 * bottom_violation
        
        return ManifoldIntegrityResult(
            mis=float(mis),
            orthogonality_violation=so3_result.orthogonality_violation,
            determinant_violation=so3_result.determinant_violation,
            is_valid=mis < self.VALIDITY_THRESHOLD,
            details={
                **so3_result.details,
                'bottom_row_violation': float(bottom_violation),
            }
        )
    
    def _compute_stiefel(self, V: np.ndarray) -> ManifoldIntegrityResult:
        """
        Compute MIS for Stiefel manifold (orthonormal frames).
        
        Stiefel constraint: V^T V = I_k (columns are orthonormal)
        """
        k = V.shape[1]
        
        # Orthonormality violation
        VtV = V.T @ V
        orth_diff = VtV - np.eye(k)
        orth_violation = np.linalg.norm(orth_diff, 'fro') / np.sqrt(k)
        
        mis = orth_violation
        
        return ManifoldIntegrityResult(
            mis=float(mis),
            orthogonality_violation=float(orth_violation),
            is_valid=mis < self.VALIDITY_THRESHOLD,
            details={
                'max_orth_deviation': float(np.abs(orth_diff).max()),
            }
        )


def compute_mis_for_imputation(original: np.ndarray,
                                imputed: np.ndarray,
                                mask: np.ndarray,
                                manifold_type: ManifoldType) -> Dict[str, float]:
    """
    Compute MIS specifically for imputation results.
    
    Returns metrics on both observed and imputed (missing) regions.
    """
    mis_calc = ManifoldIntegrityScore(manifold_type)
    
    # Overall MIS
    if imputed.ndim == 3:  # Batch of matrices
        overall_mis, overall_result = mis_calc.compute_batch(imputed)
    else:
        overall_result = mis_calc.compute(imputed)
        overall_mis = overall_result.mis
    
    return {
        'mis': overall_mis,
        'symmetry_violation': overall_result.symmetry_violation,
        'positivity_violation': overall_result.positivity_violation,
        'orthogonality_violation': overall_result.orthogonality_violation,
        'determinant_violation': overall_result.determinant_violation,
        'norm_violation': overall_result.norm_violation,
        'is_valid': overall_result.is_valid,
        'validity_rate': overall_result.details.get('validity_rate', float(overall_result.is_valid)),
    }

