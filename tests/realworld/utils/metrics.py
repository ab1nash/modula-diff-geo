"""
Standard ML Metrics for Missing Data Evaluation

Provides Hits@K, MRR, RMSE, MAE, R² for evaluating imputation quality.
"""
from typing import NamedTuple

import jax.numpy as jnp


class ImputationMetrics(NamedTuple):
    """Container for imputation evaluation metrics."""
    rmse: float                 # Root Mean Squared Error
    mae: float                  # Mean Absolute Error  
    hits_at_1: float            # Fraction of predictions in top-1
    hits_at_3: float            # Fraction of predictions in top-3
    hits_at_10: float           # Fraction of predictions in top-10
    mrr: float                  # Mean Reciprocal Rank
    r2_score: float             # R² coefficient of determination


class MissingDataEvaluator:
    """
    Evaluate missing data imputation using standard ML metrics.
    
    Provides:
    - Hits@K: For discretized values, measures if true value is in top-K predictions
    - MRR: Mean Reciprocal Rank - average of 1/rank of correct answer
    - RMSE/MAE: Standard regression metrics
    - R²: Coefficient of determination
    """
    
    @staticmethod
    def compute_all_metrics(
        true_values: jnp.ndarray,
        predicted_values: jnp.ndarray,
        n_bins: int = 100
    ) -> ImputationMetrics:
        """
        Compute all imputation metrics.
        
        Args:
            true_values: Ground truth values (1D array of missing entries)
            predicted_values: Imputed/predicted values
            n_bins: Number of bins for discretization (for Hits@K, MRR)
            
        Returns:
            ImputationMetrics with all computed values
        """
        true_flat = true_values.flatten()
        pred_flat = predicted_values.flatten()
        
        # Regression metrics
        rmse = MissingDataEvaluator.compute_rmse(true_flat, pred_flat)
        mae = MissingDataEvaluator.compute_mae(true_flat, pred_flat)
        r2 = MissingDataEvaluator.compute_r2(true_flat, pred_flat)
        
        # Ranking metrics (discretized)
        hits_1 = MissingDataEvaluator.compute_hits_at_k(true_flat, pred_flat, k=1, n_bins=n_bins)
        hits_3 = MissingDataEvaluator.compute_hits_at_k(true_flat, pred_flat, k=3, n_bins=n_bins)
        hits_10 = MissingDataEvaluator.compute_hits_at_k(true_flat, pred_flat, k=10, n_bins=n_bins)
        mrr = MissingDataEvaluator.compute_mrr(true_flat, pred_flat, n_bins=n_bins)
        
        return ImputationMetrics(
            rmse=float(rmse),
            mae=float(mae),
            hits_at_1=float(hits_1),
            hits_at_3=float(hits_3),
            hits_at_10=float(hits_10),
            mrr=float(mrr),
            r2_score=float(r2)
        )
    
    @staticmethod
    def compute_rmse(true: jnp.ndarray, pred: jnp.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(jnp.sqrt(jnp.mean((true - pred) ** 2)))
    
    @staticmethod
    def compute_mae(true: jnp.ndarray, pred: jnp.ndarray) -> float:
        """Mean Absolute Error."""
        return float(jnp.mean(jnp.abs(true - pred)))
    
    @staticmethod
    def compute_r2(true: jnp.ndarray, pred: jnp.ndarray) -> float:
        """R² coefficient of determination."""
        ss_res = jnp.sum((true - pred) ** 2)
        ss_tot = jnp.sum((true - jnp.mean(true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def compute_hits_at_k(
        true: jnp.ndarray, 
        pred: jnp.ndarray, 
        k: int,
        n_bins: int = 100
    ) -> float:
        """
        Hits@K: Fraction of predictions where true bin is in top-K closest bins.
        
        Discretizes the value range into bins, then checks if the true value's
        bin is within k bins of the predicted value's bin.
        """
        # Compute global value range
        all_vals = jnp.concatenate([true, pred])
        val_min, val_max = jnp.min(all_vals), jnp.max(all_vals)
        val_range = val_max - val_min + 1e-8
        
        # Discretize to bins
        true_bins = jnp.floor((true - val_min) / val_range * n_bins).astype(jnp.int32)
        pred_bins = jnp.floor((pred - val_min) / val_range * n_bins).astype(jnp.int32)
        
        # Clamp to valid range
        true_bins = jnp.clip(true_bins, 0, n_bins - 1)
        pred_bins = jnp.clip(pred_bins, 0, n_bins - 1)
        
        # Hit if |true_bin - pred_bin| < k
        hits = jnp.abs(true_bins - pred_bins) < k
        
        return float(jnp.mean(hits))
    
    @staticmethod
    def compute_mrr(
        true: jnp.ndarray,
        pred: jnp.ndarray,
        n_bins: int = 100
    ) -> float:
        """
        Mean Reciprocal Rank.
        
        Rank = |true_bin - pred_bin| + 1 (rank 1 = perfect match)
        MRR = mean(1/rank)
        """
        all_vals = jnp.concatenate([true, pred])
        val_min, val_max = jnp.min(all_vals), jnp.max(all_vals)
        val_range = val_max - val_min + 1e-8
        
        true_bins = jnp.floor((true - val_min) / val_range * n_bins).astype(jnp.int32)
        pred_bins = jnp.floor((pred - val_min) / val_range * n_bins).astype(jnp.int32)
        
        true_bins = jnp.clip(true_bins, 0, n_bins - 1)
        pred_bins = jnp.clip(pred_bins, 0, n_bins - 1)
        
        # Rank = distance + 1
        ranks = jnp.abs(true_bins - pred_bins) + 1
        reciprocal_ranks = 1.0 / ranks
        
        return float(jnp.mean(reciprocal_ranks))
    
    @staticmethod
    def print_metrics(metrics: ImputationMetrics, label: str = ""):
        """Pretty print all metrics."""
        prefix = f"{label}: " if label else ""
        print(f"  {prefix}RMSE={metrics.rmse:.4f}, MAE={metrics.mae:.4f}, R²={metrics.r2_score:.4f}")
        print(f"  {prefix}Hits@1={metrics.hits_at_1:.2%}, Hits@3={metrics.hits_at_3:.2%}, Hits@10={metrics.hits_at_10:.2%}")
        print(f"  {prefix}MRR={metrics.mrr:.4f}")

