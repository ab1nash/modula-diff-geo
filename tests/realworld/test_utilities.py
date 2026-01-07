"""
Utility Tests: DataMasker and MissingDataEvaluator

Unit tests for the supporting utility classes.
"""
import pytest
import jax
import jax.numpy as jnp

from .utils import DataMasker, MaskPattern, MissingDataEvaluator


@pytest.mark.hypothesis
class TestImputationMetricsUtility:
    """Unit tests for the MissingDataEvaluator metrics."""
    
    def test_perfect_prediction_metrics(self, key):
        """Test that perfect predictions give optimal metric values."""
        true_vals = jax.random.normal(key, (100,))
        pred_vals = true_vals  # Perfect prediction
        
        metrics = MissingDataEvaluator.compute_all_metrics(true_vals, pred_vals)
        
        assert metrics.rmse < 1e-6, "FAIL: RMSE should be ~0 for perfect prediction"
        assert metrics.mae < 1e-6, "FAIL: MAE should be ~0 for perfect prediction"
        assert metrics.r2_score > 0.999, "FAIL: R² should be ~1 for perfect prediction"
        assert metrics.hits_at_1 > 0.99, "FAIL: Hits@1 should be ~1 for perfect prediction"
        assert metrics.mrr > 0.99, "FAIL: MRR should be ~1 for perfect prediction"
        
        print("PASS: Perfect predictions yield optimal metric values")
    
    def test_random_prediction_metrics(self, key):
        """Test that random predictions give poor metric values."""
        k1, k2 = jax.random.split(key)
        true_vals = jax.random.normal(k1, (100,))
        pred_vals = jax.random.normal(k2, (100,)) * 5  # Random, different scale
        
        metrics = MissingDataEvaluator.compute_all_metrics(true_vals, pred_vals)
        
        assert metrics.r2_score < 0.5, "FAIL: R² should be low for random predictions"
        assert metrics.hits_at_1 < 0.5, "FAIL: Hits@1 should be low for random predictions"
        
        print(f"PASS: Random predictions yield poor metrics (R²={metrics.r2_score:.3f}, Hits@1={metrics.hits_at_1:.2%})")
    
    def test_metrics_bounds(self, key):
        """Test that metrics stay within expected bounds."""
        k1, k2 = jax.random.split(key)
        
        for _ in range(5):
            k1, k2 = jax.random.split(k1)
            true_vals = jax.random.normal(k1, (50,))
            pred_vals = jax.random.normal(k2, (50,))
            
            metrics = MissingDataEvaluator.compute_all_metrics(true_vals, pred_vals)
            
            assert metrics.rmse >= 0, "FAIL: RMSE must be non-negative"
            assert metrics.mae >= 0, "FAIL: MAE must be non-negative"
            assert 0 <= metrics.hits_at_1 <= 1, "FAIL: Hits@1 must be in [0,1]"
            assert 0 <= metrics.hits_at_3 <= 1, "FAIL: Hits@3 must be in [0,1]"
            assert 0 <= metrics.hits_at_10 <= 1, "FAIL: Hits@10 must be in [0,1]"
            assert 0 < metrics.mrr <= 1, "FAIL: MRR must be in (0,1]"
        
        print("PASS: All metric bounds respected across random trials")


@pytest.mark.hypothesis
class TestDataMaskerUtility:
    """Unit tests for the DataMasker utility class."""
    
    def test_masker_preserves_dimensions(self, key):
        """Test that masking preserves data shape."""
        shapes = [(10,), (10, 5), (10, 5, 3)]
        
        for shape in shapes:
            data = jax.random.normal(key, shape)
            masked = DataMasker.apply_mask(data, 0.3, MaskPattern.UNIFORM_RANDOM, key)
            
            assert masked.original.shape == shape, \
                f"FAIL: Original shape {masked.original.shape} != {shape}"
            assert masked.masked.shape == shape, \
                f"FAIL: Masked shape {masked.masked.shape} != {shape}"
            assert masked.mask.shape == shape, \
                f"FAIL: Mask shape {masked.mask.shape} != {shape}"
        
        print("PASS: DataMasker preserves all input shapes correctly")
    
    def test_masker_respects_fraction(self, key):
        """Test that actual missing fraction is close to requested."""
        data = jax.random.normal(key, (100, 50))
        
        for target_frac in [0.1, 0.3, 0.5, 0.7]:
            masked = DataMasker.apply_mask(data, target_frac, MaskPattern.UNIFORM_RANDOM, key)
            actual_frac = masked.missing_fraction
            
            assert abs(actual_frac - target_frac) < 0.1, \
                f"FAIL: Requested {target_frac:.0%} missing, got {actual_frac:.0%}"
        
        print("PASS: DataMasker achieves requested missing fractions within tolerance")
    
    def test_masker_sweep_returns_correct_count(self, key):
        """Test sweep function returns correct number of masked versions."""
        data = jax.random.normal(key, (20, 10))
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = DataMasker.sweep_missing_fractions(
            data, fractions, MaskPattern.UNIFORM_RANDOM, key
        )
        
        assert len(results) == len(fractions), \
            f"FAIL: Expected {len(fractions)} results, got {len(results)}"
        print("PASS: Sweep function returns correct number of masked versions")
    
    def test_all_patterns_work(self, key):
        """Test that all masking patterns execute without error."""
        data = jax.random.normal(key, (20, 10))
        
        for pattern in MaskPattern:
            try:
                masked = DataMasker.apply_mask(data, 0.3, pattern, key)
                assert masked.masked.shape == data.shape
            except Exception as e:
                assert False, f"FAIL: Pattern {pattern.value} raised {e}"
        
        print("PASS: All masking patterns execute successfully")

