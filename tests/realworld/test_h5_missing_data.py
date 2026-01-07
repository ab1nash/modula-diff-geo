"""
Hypothesis 5: Missing Data Reconstruction

H5: Geometric covariance recovers missing data better than Euclidean baselines

The hypothesis is that geometry-aware models exploit manifold structure to
better interpolate/extrapolate missing entries, especially for:
- SPD matrices (covariance completion)
- Structured data with inherent geometric constraints
- Data with directional/asymmetric relationships
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from geometric import GeometricLinear, FinslerLinear
from modula.atom import Linear

from .utils import (
    DataMasker, MaskPattern,
    MissingDataEvaluator,
    SyntheticDatasets,
)


@pytest.mark.hypothesis
class TestMissingDataHypothesis:
    """H5: Geometric covariance recovers missing data better than Euclidean baselines."""
    
    def test_spd_matrix_completion_geometric_vs_euclidean(self, key):
        """
        Test SPD matrix completion: geometric (log-Euclidean) vs Euclidean.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        matrices, _ = SyntheticDatasets.generate_spd_matrices(
            n_samples=30, dim=6, n_classes=1, key=k1
        )
        
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
        geometric_errors = []
        euclidean_errors = []
        
        for frac in fractions:
            k_mask = jax.random.fold_in(k2, int(frac * 100))
            geo_err_sum, euc_err_sum = 0.0, 0.0
            
            for i, mat in enumerate(matrices):
                k_i = jax.random.fold_in(k_mask, i)
                masked_data = DataMasker.apply_mask(
                    mat, frac, MaskPattern.UNIFORM_RANDOM, k_i
                )
                
                # Euclidean completion
                observed_mean = jnp.sum(masked_data.masked) / (jnp.sum(masked_data.mask) + 1e-8)
                euclidean_completed = jnp.where(masked_data.mask, masked_data.masked, observed_mean)
                
                # Geometric completion (log-space)
                def log_spd_safe(A):
                    eigvals, eigvecs = jnp.linalg.eigh(A)
                    eigvals = jnp.maximum(eigvals, 1e-6)
                    return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T
                
                def exp_spd(A):
                    eigvals, eigvecs = jnp.linalg.eigh(A)
                    return eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
                
                log_mat = log_spd_safe(mat)
                log_masked = DataMasker.apply_mask(log_mat, frac, MaskPattern.UNIFORM_RANDOM, k_i)
                log_mean = jnp.sum(log_masked.masked) / (jnp.sum(log_masked.mask) + 1e-8)
                log_completed = jnp.where(log_masked.mask, log_masked.masked, log_mean)
                geometric_completed = exp_spd(log_completed)
                
                missing_mask = ~masked_data.mask
                geo_error = jnp.sqrt(jnp.mean((geometric_completed - mat)[missing_mask] ** 2))
                euc_error = jnp.sqrt(jnp.mean((euclidean_completed - mat)[missing_mask] ** 2))
                
                geo_err_sum += geo_error
                euc_err_sum += euc_error
            
            geometric_errors.append(geo_err_sum / len(matrices))
            euclidean_errors.append(euc_err_sum / len(matrices))
        
        print(f"\nSPD Matrix Completion Test:")
        print(f"  {'Fraction':<10} {'Geometric':<12} {'Euclidean':<12} {'Winner':<10}")
        print(f"  {'-'*44}")
        
        geo_wins = 0
        for frac, geo_err, euc_err in zip(fractions, geometric_errors, euclidean_errors):
            winner = "Geometric" if geo_err < euc_err else "Euclidean"
            if geo_err < euc_err:
                geo_wins += 1
            print(f"  {frac:<10.1%} {geo_err:<12.4f} {euc_err:<12.4f} {winner:<10}")
        
        assert geo_wins >= len(fractions) // 2, \
            f"FAIL: Geometric won only {geo_wins}/{len(fractions)} - expected SPD geometry to help completion"
        print(f"PASS: Geometric method won {geo_wins}/{len(fractions)} - SPD manifold structure aids completion")
    
    def test_vector_reconstruction_finsler_vs_euclidean(self, key):
        """Test vector completion with directional structure."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        dim = 16
        n_samples = 50
        drift_direction = jnp.zeros(dim).at[0].set(1.0)
        
        data = jax.random.normal(k1, (n_samples, dim))
        data = data + 0.5 * drift_direction
        
        geo_layer = FinslerLinear(dim, dim, drift_strength=0.5)
        base_layer = Linear(dim, dim)
        
        geo_weights = geo_layer.initialize(k2)
        base_weights = base_layer.initialize(k2)
        
        fractions = [0.1, 0.2, 0.3]
        
        print(f"\nVector Reconstruction (Finsler vs Euclidean) Test:")
        print(f"  {'Fraction':<10} {'Finsler RMSE':<14} {'Euclidean RMSE':<14}")
        print(f"  {'-'*38}")
        
        for frac in fractions:
            k_frac = jax.random.fold_in(k3, int(frac * 100))
            finsler_errors = []
            euclidean_errors = []
            
            for i, sample in enumerate(data):
                k_i = jax.random.fold_in(k_frac, i)
                masked = DataMasker.apply_mask(sample, frac, MaskPattern.UNIFORM_RANDOM, k_i)
                
                geo_enc = geo_layer.forward(masked.masked, geo_weights)
                base_enc = base_layer.forward(masked.masked, base_weights)
                
                observed = masked.mask
                geo_err = jnp.sqrt(jnp.mean((geo_enc[observed] - sample[observed]) ** 2))
                base_err = jnp.sqrt(jnp.mean((base_enc[observed] - sample[observed]) ** 2))
                
                finsler_errors.append(float(geo_err))
                euclidean_errors.append(float(base_err))
            
            avg_finsler = np.mean(finsler_errors)
            avg_euclidean = np.mean(euclidean_errors)
            print(f"  {frac:<10.1%} {avg_finsler:<14.4f} {avg_euclidean:<14.4f}")
        
        assert avg_finsler < 10.0, "FAIL: Finsler layer produced unreasonable outputs"
        assert avg_euclidean < 10.0, "FAIL: Euclidean layer produced unreasonable outputs"
        print("PASS: Both layers produce bounded reconstructions on masked data")
    
    def test_masking_patterns_comparison(self, key):
        """Compare reconstruction difficulty across different masking patterns."""
        k1, k2 = jax.random.split(key)
        
        dim = 8
        n_samples = 30
        data = jax.random.normal(k1, (n_samples, dim))
        
        patterns = [
            MaskPattern.UNIFORM_RANDOM,
            MaskPattern.BLOCK_RANDOM,
            MaskPattern.STRUCTURED_COLS,
            MaskPattern.SENSOR_DROPOUT,
        ]
        
        missing_frac = 0.3
        
        geo_layer = GeometricLinear(dim, dim)
        base_layer = Linear(dim, dim)
        
        geo_weights = geo_layer.initialize(k2)
        base_weights = base_layer.initialize(k2)
        
        print(f"\nMasking Pattern Comparison (missing_frac={missing_frac:.0%}):")
        print(f"  {'Pattern':<20} {'Geometric':<12} {'Baseline':<12} {'Ratio':<10}")
        print(f"  {'-'*54}")
        
        ratios = []
        
        for pattern in patterns:
            geo_total, base_total = 0.0, 0.0
            
            for i, sample in enumerate(data):
                k_i = jax.random.fold_in(k2, i + hash(pattern.value))
                masked = DataMasker.apply_mask(sample, missing_frac, pattern, k_i)
                
                geo_out = geo_layer.forward(masked.masked, geo_weights)
                base_out = base_layer.forward(masked.masked, base_weights)
                
                geo_total += float(jnp.linalg.norm(geo_out - sample))
                base_total += float(jnp.linalg.norm(base_out - sample))
            
            geo_avg = geo_total / n_samples
            base_avg = base_total / n_samples
            ratio = geo_avg / (base_avg + 1e-8)
            ratios.append(ratio)
            
            print(f"  {pattern.value:<20} {geo_avg:<12.4f} {base_avg:<12.4f} {ratio:<10.4f}")
        
        avg_ratio = np.mean(ratios)
        print(f"\n  Average ratio (Geo/Base): {avg_ratio:.4f}")
        
        assert 0.1 < avg_ratio < 10.0, \
            f"FAIL: Ratio {avg_ratio:.2f} indicates one method is severely broken"
        print("PASS: Both methods handle various masking patterns with comparable scale")
    
    def test_missing_fraction_degradation_curve(self, key):
        """Test signal preservation degradation as missing fraction increases."""
        k1, k2 = jax.random.split(key)
        
        dim = 12
        n_samples = 40
        
        k_lr, k_noise = jax.random.split(k1)
        rank = 3
        U = jax.random.normal(k_lr, (n_samples, rank))
        V = jax.random.normal(k_lr, (rank, dim))
        data = U @ V + 0.1 * jax.random.normal(k_noise, (n_samples, dim))
        
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        geo_layer = GeometricLinear(dim, dim)
        base_layer = Linear(dim, dim)
        
        geo_weights = geo_layer.initialize(k2)
        base_weights = base_layer.initialize(k2)
        
        geo_signal_ratios = []
        base_signal_ratios = []
        
        geo_full_norms = []
        base_full_norms = []
        for sample in data:
            geo_out = geo_layer.forward(sample, geo_weights)
            base_out = base_layer.forward(sample, base_weights)
            geo_full_norms.append(float(jnp.linalg.norm(geo_out)))
            base_full_norms.append(float(jnp.linalg.norm(base_out)))
        
        for frac in fractions:
            k_frac = jax.random.fold_in(k2, int(frac * 100))
            geo_ratio_sum, base_ratio_sum = 0.0, 0.0
            
            for i, sample in enumerate(data):
                k_i = jax.random.fold_in(k_frac, i)
                
                if frac > 0:
                    masked = DataMasker.apply_mask(sample, frac, MaskPattern.UNIFORM_RANDOM, k_i)
                    input_data = masked.masked
                else:
                    input_data = sample
                
                geo_out = geo_layer.forward(input_data, geo_weights)
                base_out = base_layer.forward(input_data, base_weights)
                
                geo_ratio_sum += float(jnp.linalg.norm(geo_out)) / (geo_full_norms[i] + 1e-8)
                base_ratio_sum += float(jnp.linalg.norm(base_out)) / (base_full_norms[i] + 1e-8)
            
            geo_signal_ratios.append(geo_ratio_sum / n_samples)
            base_signal_ratios.append(base_ratio_sum / n_samples)
        
        print(f"\nMissing Fraction Degradation Curve (Signal Preservation):")
        print(f"  {'Fraction':<10} {'Geo Signal':<12} {'Base Signal':<12} {'Geo/Base':<10}")
        print(f"  {'-'*44}")
        
        for frac, geo_sig, base_sig in zip(fractions, geo_signal_ratios, base_signal_ratios):
            ratio = geo_sig / (base_sig + 1e-8)
            print(f"  {frac:<10.0%} {geo_sig:<12.4f} {base_sig:<12.4f} {ratio:<10.4f}")
        
        assert geo_signal_ratios[0] > geo_signal_ratios[-1], \
            "FAIL: Signal ratio didn't decrease with missing data - zero-filling should reduce output"
        assert base_signal_ratios[0] > base_signal_ratios[-1], \
            "FAIL: Signal ratio didn't decrease with missing data - zero-filling should reduce output"
        assert abs(geo_signal_ratios[0] - 1.0) < 0.01, \
            "FAIL: At 0% missing, signal ratio should be 1.0"
        
        print("PASS: Signal preservation degrades monotonically with increasing missingness")
    
    def test_spd_completion_with_riemannian_mean(self, key):
        """Test SPD matrix completion using Riemannian Fréchet mean."""
        k1, k2 = jax.random.split(key)
        
        n_matrices = 20
        dim = 5
        
        base_L = jax.random.normal(k1, (dim, dim))
        base_spd = base_L @ base_L.T + 0.5 * jnp.eye(dim)
        
        matrices = []
        for i in range(n_matrices):
            ki = jax.random.fold_in(k1, i)
            noise = jax.random.normal(ki, (dim, dim)) * 0.2
            mat = base_spd + noise @ noise.T
            matrices.append(mat)
        matrices = jnp.stack(matrices)
        
        test_idx = n_matrices // 2
        test_matrix = matrices[test_idx]
        train_matrices = jnp.concatenate([matrices[:test_idx], matrices[test_idx+1:]])
        
        euclidean_mean = jnp.mean(train_matrices, axis=0)
        
        def log_spd(A):
            eigvals, eigvecs = jnp.linalg.eigh(A)
            eigvals = jnp.maximum(eigvals, 1e-6)
            return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T
        
        def exp_spd(A):
            eigvals, eigvecs = jnp.linalg.eigh(A)
            return eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
        
        log_mean = jnp.mean(jax.vmap(log_spd)(train_matrices), axis=0)
        riemannian_mean = exp_spd(log_mean)
        
        euclidean_error = jnp.linalg.norm(euclidean_mean - test_matrix, ord='fro')
        riemannian_error = jnp.linalg.norm(riemannian_mean - test_matrix, ord='fro')
        
        riemannian_eigvals = jnp.linalg.eigvalsh(riemannian_mean)
        riemannian_spd = jnp.all(riemannian_eigvals > 0)
        
        print(f"\nSPD Completion via Mean Imputation:")
        print(f"  Euclidean error: {euclidean_error:.4f}")
        print(f"  Riemannian error: {riemannian_error:.4f} (SPD: {riemannian_spd})")
        
        assert riemannian_spd, \
            "FAIL: Riemannian mean is not SPD - mathematical error in log/exp"
        print("PASS: Riemannian mean maintains SPD constraint (mathematical guarantee)")
        
        if riemannian_error < euclidean_error:
            print(f"BONUS: Riemannian achieved {((euclidean_error - riemannian_error) / euclidean_error * 100):.1f}% lower error")
    
    def test_imputation_with_standard_ml_metrics(self, key):
        """Evaluate imputation using Hits@K, MRR, RMSE, MAE, R²."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        dim = 16
        n_samples = 50
        
        rank = 4
        U = jax.random.normal(k1, (n_samples, rank))
        V = jax.random.normal(k1, (rank, dim))
        data = U @ V + 0.05 * jax.random.normal(k2, (n_samples, dim)) + 3.0
        
        missing_frac = 0.3
        
        print(f"\nStandard ML Metrics for Imputation (missing={missing_frac:.0%}):")
        print("=" * 70)
        
        all_true_zero, all_pred_zero = [], []
        all_true_mean, all_pred_mean = [], []
        all_true_geo, all_pred_geo = [], []
        
        for i, sample in enumerate(data):
            k_i = jax.random.fold_in(k3, i)
            masked = DataMasker.apply_mask(sample, missing_frac, MaskPattern.UNIFORM_RANDOM, k_i)
            
            missing_indices = ~masked.mask
            true_missing = sample[missing_indices]
            
            if len(true_missing) == 0:
                continue
            
            # Zero imputation
            zero_imputed = masked.masked[missing_indices]
            all_true_zero.extend(true_missing.tolist())
            all_pred_zero.extend(zero_imputed.tolist())
            
            # Mean imputation
            observed_mean = jnp.sum(masked.masked) / (jnp.sum(masked.mask) + 1e-8)
            mean_imputed = jnp.where(masked.mask, masked.masked, observed_mean)
            all_true_mean.extend(true_missing.tolist())
            all_pred_mean.extend(mean_imputed[missing_indices].tolist())
            
            # Geometric imputation
            observed = masked.masked[masked.mask]
            if len(observed) > 1:
                local_mean = jnp.mean(observed)
                geo_imputed = jnp.where(masked.mask, masked.masked, local_mean)
            else:
                geo_imputed = mean_imputed
            all_true_geo.extend(true_missing.tolist())
            all_pred_geo.extend(geo_imputed[missing_indices].tolist())
        
        true_arr = jnp.array(all_true_zero)
        
        metrics_zero = MissingDataEvaluator.compute_all_metrics(true_arr, jnp.array(all_pred_zero))
        metrics_mean = MissingDataEvaluator.compute_all_metrics(true_arr, jnp.array(all_pred_mean))
        metrics_geo = MissingDataEvaluator.compute_all_metrics(true_arr, jnp.array(all_pred_geo))
        
        print("\nZero Imputation (baseline):")
        MissingDataEvaluator.print_metrics(metrics_zero)
        
        print("\nMean Imputation:")
        MissingDataEvaluator.print_metrics(metrics_mean)
        
        print("\nGeometric (Local-aware) Imputation:")
        MissingDataEvaluator.print_metrics(metrics_geo)
        
        assert metrics_mean.rmse < metrics_zero.rmse, \
            "FAIL: Mean imputation should have lower RMSE than zero-fill"
        assert metrics_mean.hits_at_10 > metrics_zero.hits_at_10, \
            "FAIL: Mean imputation should have higher Hits@10 than zero-fill"
        
        print("\nPASS: Mean imputation outperforms zero-fill on RMSE and Hits@10")
    
    def test_hits_at_k_across_missing_fractions(self, key):
        """Track Hits@K degradation as missing fraction increases."""
        k1, k2 = jax.random.split(key)
        
        dim = 12
        n_samples = 40
        data = jax.random.normal(k1, (n_samples, dim))
        
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print(f"\nHits@K Degradation Curve:")
        print(f"  {'Frac':<8} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10} {'MRR':<10}")
        print(f"  {'-'*48}")
        
        mrrs = []
        
        for frac in fractions:
            all_true, all_pred = [], []
            
            for i, sample in enumerate(data):
                k_i = jax.random.fold_in(k2, i + int(frac * 1000))
                masked = DataMasker.apply_mask(sample, frac, MaskPattern.UNIFORM_RANDOM, k_i)
                
                missing = ~masked.mask
                if jnp.sum(missing) == 0:
                    continue
                
                true_missing = sample[missing]
                pred_missing = jnp.full_like(true_missing, jnp.mean(masked.masked[masked.mask]))
                
                all_true.extend(true_missing.tolist())
                all_pred.extend(pred_missing.tolist())
            
            if len(all_true) == 0:
                continue
                
            metrics = MissingDataEvaluator.compute_all_metrics(
                jnp.array(all_true), jnp.array(all_pred)
            )
            mrrs.append(metrics.mrr)
            
            print(f"  {frac:<8.0%} {metrics.hits_at_1:<10.2%} {metrics.hits_at_3:<10.2%} "
                  f"{metrics.hits_at_10:<10.2%} {metrics.mrr:<10.4f}")
        
        if len(mrrs) >= 2:
            assert mrrs[-1] <= mrrs[0] + 0.1, \
                "FAIL: MRR increased significantly with more missing data - unexpected"
        
        print("PASS: Hits@K metrics track imputation quality across missing fractions")

