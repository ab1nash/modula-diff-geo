"""
Data Masking Utilities for Missing Data Experiments

Provides configurable masking strategies to simulate real-world
missingness patterns with a "slider" interface via missing_fraction.
"""
from enum import Enum
from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MaskPattern(Enum):
    """Strategies for introducing missing data."""
    UNIFORM_RANDOM = "uniform_random"      # Random entries, uniform probability
    BLOCK_RANDOM = "block_random"          # Random contiguous blocks
    STRUCTURED_ROWS = "structured_rows"    # Entire rows/features missing
    STRUCTURED_COLS = "structured_cols"    # Entire columns/samples missing  
    DIAGONAL_BAND = "diagonal_band"        # Missing along diagonals (for matrices)
    SENSOR_DROPOUT = "sensor_dropout"      # Simulates sensor failures (clustered)
    TEMPORAL_GAP = "temporal_gap"          # Contiguous time gaps


class MaskedData(NamedTuple):
    """Container for masked data with ground truth."""
    original: jnp.ndarray       # Full original data
    masked: jnp.ndarray         # Data with missing values (NaN or 0)
    mask: jnp.ndarray           # Boolean mask (True = observed, False = missing)
    missing_fraction: float     # Actual fraction of missing data


class DataMasker:
    """
    Utility for introducing missing data with various patterns.
    
    Provides a "slider" interface via missing_fraction parameter (0.0 to 1.0).
    Supports multiple masking strategies to simulate different real-world
    missingness patterns.
    """
    
    @staticmethod
    def apply_mask(
        data: jnp.ndarray,
        missing_fraction: float,
        pattern: MaskPattern,
        key: jax.Array,
        fill_value: float = 0.0
    ) -> MaskedData:
        """
        Apply missing data mask to input data.
        
        Args:
            data: Input data array of any shape
            missing_fraction: Fraction of data to mask (0.0 to 1.0)
            pattern: Type of missingness pattern
            key: JAX random key
            fill_value: Value to fill masked entries (default 0.0, can use jnp.nan)
            
        Returns:
            MaskedData containing original, masked data, and mask
        """
        if missing_fraction <= 0:
            return MaskedData(data, data, jnp.ones_like(data, dtype=bool), 0.0)
        if missing_fraction >= 1:
            mask = jnp.zeros_like(data, dtype=bool)
            masked = jnp.full_like(data, fill_value)
            return MaskedData(data, masked, mask, 1.0)
        
        if pattern == MaskPattern.UNIFORM_RANDOM:
            mask = DataMasker._uniform_random_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.BLOCK_RANDOM:
            mask = DataMasker._block_random_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.STRUCTURED_ROWS:
            mask = DataMasker._structured_rows_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.STRUCTURED_COLS:
            mask = DataMasker._structured_cols_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.DIAGONAL_BAND:
            mask = DataMasker._diagonal_band_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.SENSOR_DROPOUT:
            mask = DataMasker._sensor_dropout_mask(data.shape, missing_fraction, key)
        elif pattern == MaskPattern.TEMPORAL_GAP:
            mask = DataMasker._temporal_gap_mask(data.shape, missing_fraction, key)
        else:
            raise ValueError(f"Unknown mask pattern: {pattern}")
        
        masked = jnp.where(mask, data, fill_value)
        actual_missing = 1.0 - jnp.mean(mask.astype(jnp.float32))
        
        return MaskedData(data, masked, mask, float(actual_missing))
    
    @staticmethod
    def _uniform_random_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Uniformly random missing entries."""
        return jax.random.uniform(key, shape) > missing_fraction
    
    @staticmethod
    def _block_random_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Random contiguous blocks missing."""
        mask = jnp.ones(shape, dtype=bool)
        total_elements = int(np.prod(shape))
        target_missing = int(total_elements * missing_fraction)
        
        # Determine block size based on data shape
        if len(shape) >= 2:
            block_size = max(2, min(shape[-1] // 4, shape[-2] // 4, 8))
        else:
            block_size = max(2, shape[0] // 10)
        
        k1, k2 = jax.random.split(key)
        n_blocks = max(1, target_missing // (block_size ** min(2, len(shape))))
        
        flat_mask = mask.flatten()
        block_starts = jax.random.randint(k1, (n_blocks,), 0, total_elements - block_size)
        
        for start in block_starts:
            end = min(int(start) + block_size, total_elements)
            flat_mask = flat_mask.at[int(start):end].set(False)
        
        return flat_mask.reshape(shape)
    
    @staticmethod  
    def _structured_rows_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Entire rows/samples missing (simulates sensor failures)."""
        if len(shape) < 2:
            return DataMasker._uniform_random_mask(shape, missing_fraction, key)
        
        n_rows = shape[0]
        n_missing_rows = int(n_rows * missing_fraction)
        
        row_mask = jnp.ones(n_rows, dtype=bool)
        missing_indices = jax.random.permutation(key, n_rows)[:n_missing_rows]
        row_mask = row_mask.at[missing_indices].set(False)
        
        # Broadcast to full shape
        return jnp.broadcast_to(row_mask.reshape(-1, *([1] * (len(shape) - 1))), shape)
    
    @staticmethod
    def _structured_cols_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Entire columns/features missing."""
        if len(shape) < 2:
            return DataMasker._uniform_random_mask(shape, missing_fraction, key)
        
        n_cols = shape[-1]
        n_missing_cols = int(n_cols * missing_fraction)
        
        col_mask = jnp.ones(n_cols, dtype=bool)
        missing_indices = jax.random.permutation(key, n_cols)[:n_missing_cols]
        col_mask = col_mask.at[missing_indices].set(False)
        
        return jnp.broadcast_to(col_mask, shape)
    
    @staticmethod
    def _diagonal_band_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Missing along diagonals (for matrix data like covariances)."""
        if len(shape) < 2:
            return DataMasker._uniform_random_mask(shape, missing_fraction, key)
        
        mask = jnp.ones(shape, dtype=bool)
        
        # For matrix data, mask bands around diagonal
        if len(shape) >= 2:
            n = min(shape[-1], shape[-2])
            band_width = max(1, int(n * missing_fraction))
            
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    if abs(i - j) < band_width:
                        mask = mask.at[..., i, j].set(False)
        
        return mask
    
    @staticmethod
    def _sensor_dropout_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Clustered missing data (simulates sensor failures)."""
        k1, k2 = jax.random.split(key)
        
        # Start with uniform random
        base_mask = DataMasker._uniform_random_mask(shape, missing_fraction * 0.5, k1)
        
        # Add clustered dropouts
        if len(shape) >= 2:
            n_clusters = max(1, int(shape[0] * missing_fraction * 0.3))
            cluster_size = max(2, shape[-1] // 4)
            
            for i in range(n_clusters):
                ki = jax.random.fold_in(k2, i)
                k_row, k_col = jax.random.split(ki)
                row = jax.random.randint(k_row, (), 0, shape[0])
                col_start = jax.random.randint(k_col, (), 0, max(1, shape[-1] - cluster_size))
                base_mask = base_mask.at[row, col_start:col_start + cluster_size].set(False)
        
        return base_mask
    
    @staticmethod
    def _temporal_gap_mask(shape: tuple, missing_fraction: float, key: jax.Array) -> jnp.ndarray:
        """Contiguous temporal gaps (time series missingness)."""
        if len(shape) < 1:
            return jnp.ones(shape, dtype=bool)
        
        # Assume first dimension is time
        T = shape[0]
        gap_length = int(T * missing_fraction)
        
        if gap_length >= T:
            return jnp.zeros(shape, dtype=bool)
        
        start = int(jax.random.randint(key, (), 0, max(1, T - gap_length)))
        
        mask = jnp.ones(shape, dtype=bool)
        if len(shape) == 1:
            mask = mask.at[start:start + gap_length].set(False)
        else:
            mask = mask.at[start:start + gap_length, ...].set(False)
        
        return mask
    
    @staticmethod
    def sweep_missing_fractions(
        data: jnp.ndarray,
        fractions: List[float],
        pattern: MaskPattern,
        key: jax.Array
    ) -> List[MaskedData]:
        """
        Generate masked versions at multiple missing fractions.
        
        Useful for studying degradation curves.
        """
        keys = jax.random.split(key, len(fractions))
        return [
            DataMasker.apply_mask(data, frac, pattern, k)
            for frac, k in zip(fractions, keys)
        ]

