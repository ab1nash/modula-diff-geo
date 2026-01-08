"""
Imputation Methods for Geometric Data

Implements both baseline (Euclidean/modula) and geometric (diffgeo) 
imputation methods for each manifold type.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False

from .manifold_integrity import ManifoldIntegrityScore, ManifoldType


class ImputationMethod(Enum):
    """Available imputation methods."""
    # Baselines (Euclidean) - Simple
    ZERO_FILL = "zero_fill"
    MEAN_FILL = "mean_fill"
    LINEAR_INTERP = "linear_interp"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    
    # Baselines (Euclidean) - Competitive
    SVD_IMPUTE = "svd_impute"           # Low-rank matrix completion via truncated SVD
    ITERATIVE_SVD = "iterative_svd"     # Iterative SVD (SoftImpute-style)
    KNN_WEIGHTED = "knn_weighted"       # K-nearest neighbors with distance weighting
    SPLINE_INTERP = "spline_interp"     # Cubic spline interpolation for time series
    
    # Geometric (diffgeo)
    LOG_EUCLIDEAN = "log_euclidean"
    FRECHET_MEAN = "frechet_mean"
    GEODESIC_INTERP = "geodesic_interp"
    PARALLEL_TRANSPORT = "parallel_transport"
    
    # Learning-based
    MODULA_LINEAR = "modula_linear"
    DIFFGEO_LINEAR = "diffgeo_linear"
    FINSLER_LINEAR = "finsler_linear"


@dataclass
class ImputationResult:
    """Result of an imputation experiment."""
    method: ImputationMethod
    imputed_data: np.ndarray
    original_data: np.ndarray
    mask: np.ndarray  # True = observed, False = missing
    rmse: float
    mae: float
    manifold_error: Optional[float] = None  # Manifold-specific error
    mis: float = 0.0  # Manifold Integrity Score (strict metric)
    is_valid: bool = True  # Whether result stays on manifold (MIS < threshold)


class ImputationBenchmark(ABC):
    """Base class for imputation benchmarks on specific manifolds."""
    
    @abstractmethod
    def get_manifold_type(self) -> ManifoldType:
        """Return the manifold type for MIS computation."""
        pass
    
    @abstractmethod
    def get_available_methods(self) -> List[ImputationMethod]:
        """Return list of methods applicable to this manifold."""
        pass
    
    @abstractmethod
    def impute(self, data: np.ndarray, mask: np.ndarray, 
               method: ImputationMethod) -> np.ndarray:
        """Perform imputation using specified method."""
        pass
    
    @abstractmethod
    def compute_manifold_error(self, original: np.ndarray, 
                                imputed: np.ndarray,
                                mask: np.ndarray) -> float:
        """Compute manifold-specific reconstruction error."""
        pass
    
    @abstractmethod
    def validate_on_manifold(self, data: np.ndarray) -> bool:
        """Check if data lies on the manifold."""
        pass
    
    def run_comparison(self, data: np.ndarray, mask: np.ndarray,
                       methods: Optional[List[ImputationMethod]] = None) -> List[ImputationResult]:
        """Run all specified methods and compare."""
        methods = methods or self.get_available_methods()
        results = []
        
        # Initialize MIS calculator
        mis_calc = ManifoldIntegrityScore(self.get_manifold_type())
        
        for method in methods:
            try:
                imputed = self.impute(data, mask, method)
                
                # Compute errors on missing entries only
                missing = ~mask
                if np.sum(missing) == 0:
                    rmse = 0.0
                    mae = 0.0
                else:
                    diff = imputed[missing] - data[missing]
                    rmse = float(np.sqrt(np.mean(diff ** 2)))
                    mae = float(np.mean(np.abs(diff)))
                
                manifold_error = self.compute_manifold_error(data, imputed, mask)
                
                # Compute MIS (strict manifold integrity score)
                if imputed.ndim == 3:  # Batch of matrices
                    mis_value, mis_result = mis_calc.compute_batch(imputed)
                    is_valid = mis_result.is_valid
                elif imputed.ndim == 2 and self.get_manifold_type() == ManifoldType.SPHERE:
                    # Batch of vectors
                    mis_value, mis_result = mis_calc.compute_batch(imputed)
                    is_valid = mis_result.is_valid
                else:
                    mis_result = mis_calc.compute(imputed)
                    mis_value = mis_result.mis
                    is_valid = mis_result.is_valid
                
                results.append(ImputationResult(
                    method=method,
                    imputed_data=imputed,
                    original_data=data,
                    mask=mask,
                    rmse=rmse,
                    mae=mae,
                    manifold_error=manifold_error,
                    mis=mis_value,
                    is_valid=is_valid
                ))
            except Exception as e:
                print(f"Warning: {method.value} failed: {e}")
                continue
        
        return results


class SPDImputationMethods(ImputationBenchmark):
    """
    Imputation methods for SPD matrices (covariance matrices).
    
    The key insight is that SPD matrices form a Riemannian manifold,
    so Euclidean operations can break the positive-definiteness constraint.
    """
    
    def __init__(self, regularization: float = 1e-6):
        self.reg = regularization
    
    def get_manifold_type(self) -> ManifoldType:
        return ManifoldType.SPD
    
    def get_available_methods(self) -> List[ImputationMethod]:
        return [
            # Simple Euclidean baselines
            ImputationMethod.ZERO_FILL,
            ImputationMethod.MEAN_FILL,
            # Competitive Euclidean baselines
            ImputationMethod.SVD_IMPUTE,
            ImputationMethod.ITERATIVE_SVD,
            ImputationMethod.KNN_WEIGHTED,
            # Geometric methods
            ImputationMethod.LOG_EUCLIDEAN,
            ImputationMethod.FRECHET_MEAN,
            ImputationMethod.GEODESIC_INTERP,
        ]
    
    def impute(self, data: np.ndarray, mask: np.ndarray, 
               method: ImputationMethod) -> np.ndarray:
        """
        Impute missing entries in SPD matrix/matrices.
        
        Args:
            data: SPD matrix (n, n) or batch (batch, n, n)
            mask: Boolean mask, True = observed
            method: Imputation method to use
        """
        if method == ImputationMethod.ZERO_FILL:
            return self._zero_fill(data, mask)
        elif method == ImputationMethod.MEAN_FILL:
            return self._mean_fill(data, mask)
        elif method == ImputationMethod.SVD_IMPUTE:
            return self._svd_impute(data, mask)
        elif method == ImputationMethod.ITERATIVE_SVD:
            return self._iterative_svd(data, mask)
        elif method == ImputationMethod.KNN_WEIGHTED:
            return self._knn_weighted(data, mask)
        elif method == ImputationMethod.LOG_EUCLIDEAN:
            return self._log_euclidean_fill(data, mask)
        elif method == ImputationMethod.FRECHET_MEAN:
            return self._frechet_mean_fill(data, mask)
        elif method == ImputationMethod.GEODESIC_INTERP:
            return self._geodesic_interp(data, mask)
        else:
            raise ValueError(f"Unknown method for SPD: {method}")
    
    def _zero_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill missing with zeros (bad for SPD!)."""
        result = np.where(mask, data, 0.0)
        # Add regularization to maintain positive-definiteness
        if data.ndim == 2:
            result = result + self.reg * np.eye(data.shape[0])
        else:
            for i in range(len(result)):
                result[i] = result[i] + self.reg * np.eye(data.shape[-1])
        return result
    
    def _mean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill missing with observed mean."""
        if data.ndim == 2:
            observed_mean = np.sum(data * mask) / (np.sum(mask) + 1e-8)
            result = np.where(mask, data, observed_mean)
            return self._project_to_spd(result)
        else:
            # Batch case: fill each matrix independently
            result = data.copy()
            for i in range(len(data)):
                obs_mean = np.sum(data[i] * mask[i]) / (np.sum(mask[i]) + 1e-8)
                result[i] = np.where(mask[i], data[i], obs_mean)
                result[i] = self._project_to_spd(result[i])
            return result
    
    def _svd_impute(self, data: np.ndarray, mask: np.ndarray, rank: int = None) -> np.ndarray:
        """
        Low-rank matrix completion via truncated SVD.
        
        Competitive Euclidean baseline that exploits low-rank structure.
        For SPD matrices, this can be effective when matrices have low effective rank.
        """
        if data.ndim == 2:
            # Single matrix: use truncated SVD
            n = data.shape[0]
            rank = rank or max(1, n // 2)
            
            # Initialize missing with mean
            observed_mean = np.sum(data * mask) / (np.sum(mask) + 1e-8)
            filled = np.where(mask, data, observed_mean)
            
            # SVD and truncate
            U, s, Vt = np.linalg.svd(filled, full_matrices=False)
            s_truncated = np.zeros_like(s)
            s_truncated[:rank] = s[:rank]
            result = U @ np.diag(s_truncated) @ Vt
            
            return self._project_to_spd(result)
        else:
            # Batch case
            result = data.copy()
            for i in range(len(data)):
                result[i] = self._svd_impute(data[i], mask[i], rank)
            return result
    
    def _iterative_svd(self, data: np.ndarray, mask: np.ndarray, 
                       rank: int = None, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
        """
        Iterative SVD imputation (SoftImpute-style).
        
        Competitive Euclidean baseline using iterative low-rank approximation.
        Converges to a low-rank matrix that matches observed entries.
        """
        if data.ndim == 2:
            n = data.shape[0]
            rank = rank or max(1, n // 2)
            
            # Initialize with mean
            observed_mean = np.sum(data * mask) / (np.sum(mask) + 1e-8)
            X = np.where(mask, data, observed_mean)
            
            for _ in range(max_iter):
                X_old = X.copy()
                
                # SVD step
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                s_truncated = np.zeros_like(s)
                s_truncated[:rank] = s[:rank]
                X_svd = U @ np.diag(s_truncated) @ Vt
                
                # Replace observed values with original
                X = np.where(mask, data, X_svd)
                
                # Check convergence
                if np.linalg.norm(X - X_old, 'fro') < tol * np.linalg.norm(X_old, 'fro'):
                    break
            
            return self._project_to_spd(X)
        else:
            result = data.copy()
            for i in range(len(data)):
                result[i] = self._iterative_svd(data[i], mask[i], rank, max_iter, tol)
            return result
    
    def _knn_weighted(self, data: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
        """
        K-nearest neighbors imputation with distance weighting.
        
        Competitive Euclidean baseline using local structure.
        For batch data, uses similarity between matrices to impute.
        """
        if data.ndim == 2:
            # Single matrix: use spatial KNN within the matrix
            result = data.copy()
            n = data.shape[0]
            
            for i in range(n):
                for j in range(n):
                    if mask[i, j]:
                        continue
                    
                    # Find k nearest observed neighbors
                    distances = []
                    values = []
                    
                    for di in range(-k, k+1):
                        for dj in range(-k, k+1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < n and mask[ni, nj]:
                                dist = np.sqrt(di**2 + dj**2)
                                if dist > 0:
                                    distances.append(dist)
                                    values.append(data[ni, nj])
                    
                    if distances:
                        # Inverse distance weighting
                        weights = [1.0 / d for d in distances]
                        total_weight = sum(weights)
                        result[i, j] = sum(w * v for w, v in zip(weights, values)) / total_weight
                    else:
                        # Fallback to mean
                        result[i, j] = np.sum(data * mask) / (np.sum(mask) + 1e-8)
            
            return self._project_to_spd(result)
        else:
            # Batch case: use similarity between matrices
            result = data.copy()
            n_matrices = len(data)
            
            # Compute pairwise Frobenius distances between matrices
            distances = np.zeros((n_matrices, n_matrices))
            for i in range(n_matrices):
                for j in range(n_matrices):
                    if i != j:
                        # Compare observed parts
                        common_mask = mask[i] & mask[j]
                        if np.sum(common_mask) > 0:
                            diff = data[i][common_mask] - data[j][common_mask]
                            distances[i, j] = np.sqrt(np.mean(diff**2))
                        else:
                            distances[i, j] = float('inf')
            
            for i in range(n_matrices):
                if mask[i].all():
                    continue
                
                # Find k nearest neighbors
                dists = distances[i].copy()
                dists[i] = float('inf')
                neighbor_idx = np.argsort(dists)[:k]
                neighbor_dists = dists[neighbor_idx]
                
                # Filter out infinite distances
                valid = neighbor_dists < float('inf')
                if not np.any(valid):
                    result[i] = self._mean_fill(data[i:i+1], mask[i:i+1])[0]
                    continue
                
                neighbor_idx = neighbor_idx[valid]
                neighbor_dists = neighbor_dists[valid]
                
                # Inverse distance weighted average for missing entries
                weights = 1.0 / (neighbor_dists + 1e-8)
                weights = weights / weights.sum()
                
                for idx, w in zip(neighbor_idx, weights):
                    result[i] = np.where(mask[i], data[i], 
                                         result[i] + w * np.where(~mask[i], data[idx], 0))
                
                result[i] = self._project_to_spd(result[i])
            
            return result
    
    def _log_euclidean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Log-Euclidean imputation: operate in log space.
        
        This is the key geometric method: work in the tangent space
        where SPD structure is linearized.
        """
        # Transform to log space
        log_data = self._matrix_log(data)
        
        # Fill in log space
        observed_mean = np.sum(log_data * mask) / (np.sum(mask) + 1e-8)
        log_filled = np.where(mask, log_data, observed_mean)
        
        # Transform back
        return self._matrix_exp(log_filled)
    
    def _frechet_mean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Use Fréchet mean for imputation.
        
        For a batch of matrices, compute the Riemannian mean and use it
        to fill missing entries.
        """
        if data.ndim == 2:
            return self._log_euclidean_fill(data, mask)
        
        # Compute Fréchet mean of observed matrices
        valid_matrices = [data[i] for i in range(len(data)) if mask[i].any()]
        if not valid_matrices:
            return self._zero_fill(data, mask)
        
        frechet_mean = self._compute_frechet_mean(valid_matrices)
        
        # Fill missing matrices with Fréchet mean
        result = data.copy()
        for i in range(len(data)):
            if not mask[i].all():
                # Interpolate between observed parts and Fréchet mean
                result[i] = np.where(mask[i], data[i], frechet_mean)
                result[i] = self._project_to_spd(result[i])
        
        return result
    
    def _geodesic_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Geodesic interpolation between observed matrices.
        
        For time-series SPD data, interpolate along geodesics.
        """
        if data.ndim == 2:
            return self._log_euclidean_fill(data, mask)
        
        result = data.copy()
        n = len(data)
        
        # Find observed indices
        observed_idx = [i for i in range(n) if mask[i].all()]
        
        if len(observed_idx) < 2:
            return self._log_euclidean_fill(data, mask)
        
        # Interpolate between observed matrices
        for i in range(n):
            if mask[i].all():
                continue
            
            # Find bracketing observed matrices
            prev_idx = max([j for j in observed_idx if j < i], default=observed_idx[0])
            next_idx = min([j for j in observed_idx if j > i], default=observed_idx[-1])
            
            if prev_idx == next_idx:
                result[i] = data[prev_idx]
            else:
                # Interpolate along geodesic
                t = (i - prev_idx) / (next_idx - prev_idx)
                result[i] = self._geodesic(data[prev_idx], data[next_idx], t)
        
        return result
    
    def _matrix_log(self, A: np.ndarray) -> np.ndarray:
        """Matrix logarithm for SPD matrix."""
        if A.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
        else:
            return np.array([self._matrix_log(m) for m in A])
    
    def _matrix_exp(self, A: np.ndarray) -> np.ndarray:
        """Matrix exponential for symmetric matrix."""
        if A.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(A)
            return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
        else:
            return np.array([self._matrix_exp(m) for m in A])
    
    def _matrix_sqrt(self, A: np.ndarray) -> np.ndarray:
        """Matrix square root for SPD matrix."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    def _matrix_sqrt_inv(self, A: np.ndarray) -> np.ndarray:
        """Inverse matrix square root."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    def _geodesic(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Compute point on geodesic from A to B at time t."""
        A_sqrt = self._matrix_sqrt(A)
        A_inv_sqrt = self._matrix_sqrt_inv(A)
        
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        eigvals, eigvecs = np.linalg.eigh(inner)
        eigvals = np.maximum(eigvals, 1e-10)
        inner_power_t = eigvecs @ np.diag(eigvals ** t) @ eigvecs.T
        
        return A_sqrt @ inner_power_t @ A_sqrt
    
    def _compute_frechet_mean(self, matrices: List[np.ndarray], 
                              max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
        """Compute Fréchet mean via iterative algorithm."""
        # Initialize with Euclidean mean
        M = np.mean(matrices, axis=0)
        M = self._project_to_spd(M)
        
        for _ in range(max_iter):
            # Compute weighted average in tangent space
            tangent_sum = np.zeros_like(M)
            for P in matrices:
                # Log map at M
                M_sqrt = self._matrix_sqrt(M)
                M_inv_sqrt = self._matrix_sqrt_inv(M)
                inner = M_inv_sqrt @ P @ M_inv_sqrt
                log_inner = self._matrix_log(inner)
                tangent = M_sqrt @ log_inner @ M_sqrt
                tangent_sum += tangent
            
            tangent_mean = tangent_sum / len(matrices)
            
            if np.linalg.norm(tangent_mean, 'fro') < tol:
                break
            
            # Exp map to update M
            M_sqrt = self._matrix_sqrt(M)
            M_inv_sqrt = self._matrix_sqrt_inv(M)
            inner = M_inv_sqrt @ tangent_mean @ M_inv_sqrt
            exp_inner = self._matrix_exp(inner)
            M = M_sqrt @ exp_inner @ M_sqrt
        
        return M
    
    def _project_to_spd(self, A: np.ndarray) -> np.ndarray:
        """Project matrix to SPD cone."""
        A_sym = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        eigvals = np.maximum(eigvals, self.reg)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def compute_manifold_error(self, original: np.ndarray, 
                                imputed: np.ndarray,
                                mask: np.ndarray) -> float:
        """Compute Riemannian distance error."""
        if original.ndim == 2:
            return self._riemannian_distance(original, imputed)
        else:
            distances = []
            for i in range(len(original)):
                if not mask[i].all():
                    distances.append(self._riemannian_distance(original[i], imputed[i]))
            return float(np.mean(distances)) if distances else 0.0
    
    def _riemannian_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """Affine-invariant Riemannian distance."""
        A_inv_sqrt = self._matrix_sqrt_inv(A)
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        log_inner = self._matrix_log(inner)
        return float(np.linalg.norm(log_inner, 'fro'))
    
    def validate_on_manifold(self, data: np.ndarray) -> bool:
        """Check if matrix is SPD."""
        if data.ndim == 2:
            eigvals = np.linalg.eigvalsh(data)
            return bool(np.all(eigvals > 0) and np.allclose(data, data.T))
        else:
            return all(self.validate_on_manifold(m) for m in data)


class SphericalImputationMethods(ImputationBenchmark):
    """
    Imputation methods for spherical (S²) data.
    
    Used for geospatial data where coordinates are lat/lon on Earth.
    """
    
    def get_manifold_type(self) -> ManifoldType:
        return ManifoldType.SPHERE
    
    def get_available_methods(self) -> List[ImputationMethod]:
        return [
            # Simple Euclidean baselines
            ImputationMethod.ZERO_FILL,
            ImputationMethod.MEAN_FILL,
            ImputationMethod.NEAREST_NEIGHBOR,
            # Competitive Euclidean baselines
            ImputationMethod.KNN_WEIGHTED,
            ImputationMethod.SPLINE_INTERP,
            # Geometric methods
            ImputationMethod.GEODESIC_INTERP,
        ]
    
    def impute(self, data: np.ndarray, mask: np.ndarray, 
               method: ImputationMethod) -> np.ndarray:
        """
        Impute missing values in spherical data.
        
        Args:
            data: (n_points, 2 + n_features) where [:, :2] are lat/lon in radians
            mask: Boolean mask
        """
        if method == ImputationMethod.ZERO_FILL:
            return np.where(mask, data, 0.0)
        elif method == ImputationMethod.MEAN_FILL:
            return self._mean_fill(data, mask)
        elif method == ImputationMethod.NEAREST_NEIGHBOR:
            return self._nearest_neighbor(data, mask)
        elif method == ImputationMethod.KNN_WEIGHTED:
            return self._knn_weighted(data, mask)
        elif method == ImputationMethod.SPLINE_INTERP:
            return self._spline_interp(data, mask)
        elif method == ImputationMethod.GEODESIC_INTERP:
            return self._geodesic_interp(data, mask)
        else:
            raise ValueError(f"Unknown method for Spherical: {method}")
    
    def _mean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill with spherical mean (Fréchet mean on S²)."""
        result = data.copy()
        
        # For coordinates (first 2 columns), use spherical mean
        coords = data[:, :2]  # lat, lon
        coords_mask = mask[:, :2]
        
        # Convert to Cartesian for mean computation
        lat, lon = coords[:, 0], coords[:, 1]
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        
        # Compute mean of observed points
        obs_mask = coords_mask[:, 0]
        mean_x = np.mean(x[obs_mask])
        mean_y = np.mean(y[obs_mask])
        mean_z = np.mean(z[obs_mask])
        
        # Normalize to sphere
        norm = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
        mean_x, mean_y, mean_z = mean_x/norm, mean_y/norm, mean_z/norm
        
        # Convert back to spherical
        mean_lat = np.arcsin(mean_z)
        mean_lon = np.arctan2(mean_y, mean_x)
        
        # Fill missing coordinates
        result[:, 0] = np.where(coords_mask[:, 0], data[:, 0], mean_lat)
        result[:, 1] = np.where(coords_mask[:, 1], data[:, 1], mean_lon)
        
        # For scalar fields, use regular mean
        for j in range(2, data.shape[1]):
            obs_mean = np.mean(data[mask[:, j], j])
            result[:, j] = np.where(mask[:, j], data[:, j], obs_mean)
        
        return result
    
    def _nearest_neighbor(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill with nearest neighbor on sphere."""
        result = data.copy()
        n_points = len(data)
        
        for i in range(n_points):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    continue
                
                if j < 2:  # Coordinate
                    # Find nearest observed point using great circle distance
                    min_dist = float('inf')
                    nearest_val = 0.0
                    
                    for k in range(n_points):
                        if k != i and mask[k, j]:
                            dist = self._great_circle_distance(
                                data[i, 0], data[i, 1],
                                data[k, 0], data[k, 1]
                            )
                            if dist < min_dist:
                                min_dist = dist
                                nearest_val = data[k, j]
                    
                    result[i, j] = nearest_val
                else:  # Scalar field
                    # Use spatially nearest observed value
                    min_dist = float('inf')
                    nearest_val = 0.0
                    
                    for k in range(n_points):
                        if k != i and mask[k, j]:
                            dist = self._great_circle_distance(
                                data[i, 0], data[i, 1],
                                data[k, 0], data[k, 1]
                            )
                            if dist < min_dist:
                                min_dist = dist
                                nearest_val = data[k, j]
                    
                    result[i, j] = nearest_val
        
        return result
    
    def _knn_weighted(self, data: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
        """
        K-nearest neighbors with inverse distance weighting using Euclidean distance.
        
        Competitive Euclidean baseline that ignores spherical geometry.
        """
        result = data.copy()
        n_points = len(data)
        
        for i in range(n_points):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    continue
                
                # Compute Euclidean distances to all observed points
                distances = []
                values = []
                
                for idx in range(n_points):
                    if idx != i and mask[idx, j]:
                        # Use Euclidean distance (ignoring spherical geometry)
                        dist = np.sqrt(np.sum((data[i, :2] - data[idx, :2])**2))
                        if dist > 1e-10:
                            distances.append(dist)
                            values.append(data[idx, j])
                
                if len(distances) >= k:
                    # Take k nearest
                    sorted_idx = np.argsort(distances)[:k]
                    k_distances = [distances[idx] for idx in sorted_idx]
                    k_values = [values[idx] for idx in sorted_idx]
                    
                    # Inverse distance weighting
                    weights = [1.0 / d for d in k_distances]
                    total_weight = sum(weights)
                    result[i, j] = sum(w * v for w, v in zip(weights, k_values)) / total_weight
                elif distances:
                    # Use all available if less than k
                    weights = [1.0 / d for d in distances]
                    total_weight = sum(weights)
                    result[i, j] = sum(w * v for w, v in zip(weights, values)) / total_weight
                else:
                    # Fallback to mean
                    result[i, j] = np.mean(data[mask[:, j], j])
        
        return result
    
    def _spline_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Cubic spline interpolation treating data as ordered sequence.
        
        Competitive Euclidean baseline for sequential/time-series-like data.
        """
        from scipy import interpolate
        
        result = data.copy()
        n_points = len(data)
        
        for j in range(data.shape[1]):
            observed_idx = np.where(mask[:, j])[0]
            
            if len(observed_idx) < 2:
                # Not enough points for spline, fallback to mean
                if len(observed_idx) == 1:
                    result[:, j] = np.where(mask[:, j], data[:, j], data[observed_idx[0], j])
                continue
            
            observed_values = data[observed_idx, j]
            
            # Create cubic spline (or linear if only 2 points)
            if len(observed_idx) >= 4:
                kind = 'cubic'
            elif len(observed_idx) >= 2:
                kind = 'linear'
            else:
                continue
            
            try:
                f = interpolate.interp1d(observed_idx, observed_values, kind=kind,
                                        fill_value='extrapolate', bounds_error=False)
                
                # Impute missing values
                missing_idx = np.where(~mask[:, j])[0]
                if len(missing_idx) > 0:
                    result[missing_idx, j] = f(missing_idx)
            except Exception:
                # Fallback to linear interpolation
                for i in range(n_points):
                    if not mask[i, j]:
                        prev_idx = observed_idx[observed_idx < i]
                        next_idx = observed_idx[observed_idx > i]
                        
                        if len(prev_idx) == 0:
                            result[i, j] = data[next_idx[0], j]
                        elif len(next_idx) == 0:
                            result[i, j] = data[prev_idx[-1], j]
                        else:
                            p, n = prev_idx[-1], next_idx[0]
                            t = (i - p) / (n - p)
                            result[i, j] = (1 - t) * data[p, j] + t * data[n, j]
        
        return result
    
    def _geodesic_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Interpolate using inverse distance weighting on sphere."""
        result = data.copy()
        n_points = len(data)
        
        for i in range(n_points):
            for j in range(data.shape[1]):
                if mask[i, j]:
                    continue
                
                # Compute inverse distance weighted interpolation
                weights = []
                values = []
                
                for k in range(n_points):
                    if k != i and mask[k, j]:
                        dist = self._great_circle_distance(
                            data[i, 0], data[i, 1],
                            data[k, 0], data[k, 1]
                        )
                        if dist > 1e-10:
                            weights.append(1.0 / dist)
                            values.append(data[k, j])
                
                if weights:
                    total_weight = sum(weights)
                    result[i, j] = sum(w * v for w, v in zip(weights, values)) / total_weight
        
        return result
    
    def _great_circle_distance(self, lat1: float, lon1: float, 
                                lat2: float, lon2: float) -> float:
        """Compute great circle distance on unit sphere."""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * np.arcsin(np.sqrt(a))
    
    def compute_manifold_error(self, original: np.ndarray, 
                                imputed: np.ndarray,
                                mask: np.ndarray) -> float:
        """Compute error using great circle distance for coordinates."""
        errors = []
        
        for i in range(len(original)):
            if not mask[i, :2].all():  # Missing coordinates
                dist = self._great_circle_distance(
                    original[i, 0], original[i, 1],
                    imputed[i, 0], imputed[i, 1]
                )
                errors.append(dist)
        
        return float(np.mean(errors)) if errors else 0.0
    
    def validate_on_manifold(self, data: np.ndarray) -> bool:
        """Check if coordinates are valid spherical coordinates."""
        lat = data[:, 0]
        lon = data[:, 1]
        
        lat_valid = np.all(np.abs(lat) <= np.pi/2)
        lon_valid = np.all(np.abs(lon) <= np.pi)
        
        return bool(lat_valid and lon_valid)


class MocapImputationMethods(ImputationBenchmark):
    """
    Imputation methods for motion capture data.
    
    Joint angles lie on SO(3)^k (product of rotation groups).
    """
    
    def get_manifold_type(self) -> ManifoldType:
        return ManifoldType.SO3
    
    def get_available_methods(self) -> List[ImputationMethod]:
        return [
            # Simple Euclidean baselines
            ImputationMethod.ZERO_FILL,
            ImputationMethod.MEAN_FILL,
            ImputationMethod.LINEAR_INTERP,
            # Competitive Euclidean baselines
            ImputationMethod.SPLINE_INTERP,
            ImputationMethod.KNN_WEIGHTED,
            # Geometric methods
            ImputationMethod.GEODESIC_INTERP,
        ]
    
    def impute(self, data: np.ndarray, mask: np.ndarray, 
               method: ImputationMethod) -> np.ndarray:
        """
        Impute missing joint angles.
        
        Args:
            data: (n_frames, n_joints, 3) Euler angles in radians
            mask: Boolean mask
        """
        if method == ImputationMethod.ZERO_FILL:
            return np.where(mask, data, 0.0)
        elif method == ImputationMethod.MEAN_FILL:
            return self._mean_fill(data, mask)
        elif method == ImputationMethod.LINEAR_INTERP:
            return self._linear_interp(data, mask)
        elif method == ImputationMethod.SPLINE_INTERP:
            return self._spline_interp(data, mask)
        elif method == ImputationMethod.KNN_WEIGHTED:
            return self._knn_weighted(data, mask)
        elif method == ImputationMethod.GEODESIC_INTERP:
            return self._geodesic_interp(data, mask)
        else:
            raise ValueError(f"Unknown method for Mocap: {method}")
    
    def _mean_fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill with mean angle (circular mean for angles)."""
        result = data.copy()
        
        for j in range(data.shape[1]):  # joints
            for k in range(data.shape[2]):  # angle components
                angles = data[:, j, k]
                angle_mask = mask[:, j, k]
                
                if np.sum(angle_mask) == 0:
                    continue
                
                # Circular mean
                sin_mean = np.mean(np.sin(angles[angle_mask]))
                cos_mean = np.mean(np.cos(angles[angle_mask]))
                mean_angle = np.arctan2(sin_mean, cos_mean)
                
                result[:, j, k] = np.where(angle_mask, data[:, j, k], mean_angle)
        
        return result
    
    def _linear_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Linear interpolation in Euclidean space (ignores rotation structure)."""
        result = data.copy()
        n_frames = len(data)
        
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                observed_idx = np.where(mask[:, j, k])[0]
                
                if len(observed_idx) < 2:
                    continue
                
                for i in range(n_frames):
                    if mask[i, j, k]:
                        continue
                    
                    # Find bracketing observed frames
                    prev_idx = observed_idx[observed_idx < i]
                    next_idx = observed_idx[observed_idx > i]
                    
                    if len(prev_idx) == 0:
                        result[i, j, k] = data[next_idx[0], j, k]
                    elif len(next_idx) == 0:
                        result[i, j, k] = data[prev_idx[-1], j, k]
                    else:
                        p = prev_idx[-1]
                        n = next_idx[0]
                        t = (i - p) / (n - p)
                        result[i, j, k] = (1 - t) * data[p, j, k] + t * data[n, j, k]
        
        return result
    
    def _spline_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Cubic spline interpolation for time series.
        
        Competitive Euclidean baseline that captures smooth motion dynamics.
        """
        from scipy import interpolate
        
        result = data.copy()
        n_frames = len(data)
        
        for j in range(data.shape[1]):  # joints
            for k in range(data.shape[2]):  # angle components
                observed_idx = np.where(mask[:, j, k])[0]
                
                if len(observed_idx) < 2:
                    continue
                
                observed_values = data[observed_idx, j, k]
                
                # Create spline
                if len(observed_idx) >= 4:
                    kind = 'cubic'
                else:
                    kind = 'linear'
                
                try:
                    f = interpolate.interp1d(observed_idx, observed_values, kind=kind,
                                            fill_value='extrapolate', bounds_error=False)
                    
                    # Impute missing values
                    missing_idx = np.where(~mask[:, j, k])[0]
                    if len(missing_idx) > 0:
                        result[missing_idx, j, k] = f(missing_idx)
                except Exception:
                    # Fallback to linear
                    pass
        
        return result
    
    def _knn_weighted(self, data: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
        """
        K-nearest neighbors with temporal weighting.
        
        Competitive Euclidean baseline using temporal neighbors.
        """
        result = data.copy()
        n_frames = len(data)
        
        for j in range(data.shape[1]):  # joints
            for kk in range(data.shape[2]):  # angle components
                observed_idx = np.where(mask[:, j, kk])[0]
                
                if len(observed_idx) == 0:
                    continue
                
                for i in range(n_frames):
                    if mask[i, j, kk]:
                        continue
                    
                    # Find k nearest temporal neighbors
                    distances = np.abs(observed_idx - i)
                    sorted_idx = np.argsort(distances)[:k]
                    
                    k_idx = observed_idx[sorted_idx]
                    k_distances = distances[sorted_idx]
                    
                    # Inverse temporal distance weighting
                    weights = 1.0 / (k_distances + 1.0)  # +1 to avoid division by zero
                    weights = weights / weights.sum()
                    
                    result[i, j, kk] = np.sum(weights * data[k_idx, j, kk])
        
        return result
    
    def _geodesic_interp(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Spherical interpolation for rotation angles.
        
        For angles, use SLERP-like interpolation that respects periodicity.
        """
        result = data.copy()
        n_frames = len(data)
        
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                observed_idx = np.where(mask[:, j, k])[0]
                
                if len(observed_idx) < 2:
                    continue
                
                for i in range(n_frames):
                    if mask[i, j, k]:
                        continue
                    
                    prev_idx = observed_idx[observed_idx < i]
                    next_idx = observed_idx[observed_idx > i]
                    
                    if len(prev_idx) == 0:
                        result[i, j, k] = data[next_idx[0], j, k]
                    elif len(next_idx) == 0:
                        result[i, j, k] = data[prev_idx[-1], j, k]
                    else:
                        p = prev_idx[-1]
                        n = next_idx[0]
                        t = (i - p) / (n - p)
                        
                        # Shortest path interpolation on circle
                        angle_p = data[p, j, k]
                        angle_n = data[n, j, k]
                        
                        # Handle wraparound
                        diff = angle_n - angle_p
                        if diff > np.pi:
                            diff -= 2 * np.pi
                        elif diff < -np.pi:
                            diff += 2 * np.pi
                        
                        result[i, j, k] = angle_p + t * diff
        
        return result
    
    def compute_manifold_error(self, original: np.ndarray, 
                                imputed: np.ndarray,
                                mask: np.ndarray) -> float:
        """Compute angular error on SO(3)."""
        errors = []
        
        for i in range(len(original)):
            for j in range(original.shape[1]):
                if not mask[i, j].all():
                    # Compute rotation matrix error would be ideal
                    # For simplicity, use angular difference
                    for k in range(original.shape[2]):
                        diff = original[i, j, k] - imputed[i, j, k]
                        # Normalize to [-pi, pi]
                        diff = np.arctan2(np.sin(diff), np.cos(diff))
                        errors.append(np.abs(diff))
        
        return float(np.mean(errors)) if errors else 0.0
    
    def validate_on_manifold(self, data: np.ndarray) -> bool:
        """Check if angles are in valid range."""
        return bool(np.all(np.abs(data) <= 2 * np.pi))

