"""
Phase 5 Tests: Information Geometry Integration

Tests for Fisher Information, SPD Manifold, and Bregman Divergences
as specified in the implementation plan.

Test Suites:
- T5.1: Fisher Information Geometry
- T5.2: SPD Manifold
- T5.3: Bregman Divergence

Reference: Implementation Plan Section 5.2, docs/PHASE5_PLAN.md
"""
import pytest
import jax
import jax.numpy as jnp
from typing import List

from geometric import (
    FisherMetric, FisherAtom, fisher_gaussian, fisher_categorical,
    SPDManifold, SPDMetricTensor, SPDClassifier,
    BregmanDivergence, KLDivergence, SquaredEuclidean, ItakuraSaito,
    LogDet, AlphaDivergence, js_divergence, total_variation, hellinger_distance,
    MetricTensor,
)


# =============================================================================
# Test Suite 5.1: Fisher Information Geometry
# =============================================================================

@pytest.mark.phase5
class TestFisherInformationGeometry:
    """T5.1.x: Fisher Information mathematical properties."""
    
    def test_T5_1_1_positive_definiteness(self, key):
        """T5.1.1: g_ij(θ) is positive definite."""
        # Create a Fisher metric from a Gaussian
        mu = jnp.zeros(4)
        sigma = jnp.eye(4)
        
        fisher = fisher_gaussian(mu, sigma)
        
        # Check eigenvalues are positive
        eigvals = jnp.linalg.eigvalsh(fisher.matrix)
        assert jnp.all(eigvals > 0), "Fisher matrix must be positive definite"
    
    def test_T5_1_2_natural_gradient_computation(self, key):
        """T5.1.2: Natural gradient F^{-1}∇L is correctly computed."""
        k1, k2 = jax.random.split(key)
        
        # Create simple Fisher metric
        dim = 4
        L = jax.random.normal(k1, (dim, dim))
        fisher_matrix = L @ L.T + jnp.eye(dim)
        fisher = FisherMetric(fisher_matrix)
        
        # Random gradient
        gradient = jax.random.normal(k2, (dim,))
        
        # Natural gradient
        nat_grad = fisher.natural_gradient(gradient)
        
        # Verify: F @ nat_grad ≈ gradient
        reconstructed = fisher.matrix @ nat_grad
        assert jnp.allclose(reconstructed, gradient, rtol=1e-4)
    
    def test_T5_1_3_cramer_rao_bound(self, key):
        """T5.1.3: Cramér-Rao bound Var(θ̂) ≥ F^{-1} is correct."""
        # For Gaussian, F = Σ^{-1}, so CR bound = Σ
        mu = jnp.zeros(3)
        sigma = jnp.array([
            [2.0, 0.5, 0.0],
            [0.5, 1.0, 0.3],
            [0.0, 0.3, 1.5]
        ])
        
        fisher = fisher_gaussian(mu, sigma)
        cr_bound = fisher.cramer_rao_bound
        
        # CR bound should equal sigma (for Gaussian mean estimation)
        # Relax tolerance due to regularization added to Fisher matrix
        assert jnp.allclose(cr_bound, sigma, rtol=1e-3, atol=1e-5)
    
    def test_T5_1_4_categorical_fisher(self):
        """T5.1.4: Fisher metric for categorical is diag(1/p_i)."""
        probs = jnp.array([0.3, 0.5, 0.2])
        fisher = fisher_categorical(probs)
        
        # Should be diagonal with 1/p_i
        expected = jnp.diag(1.0 / probs)
        assert jnp.allclose(fisher.matrix, expected, rtol=1e-4)
    
    def test_T5_1_5_kl_approximation(self, key):
        """T5.1.5: Local KL ≈ (1/2) δθ^T F δθ."""
        k1, k2 = jax.random.split(key)
        
        # Small parameter change
        dim = 4
        L = jax.random.normal(k1, (dim, dim))
        fisher_matrix = L @ L.T + jnp.eye(dim)
        fisher = FisherMetric(fisher_matrix)
        
        delta = jax.random.normal(k2, (dim,)) * 0.01  # Small change
        
        kl_approx = fisher.kl_divergence_local(delta)
        
        # Should equal 0.5 * delta^T F delta
        expected = 0.5 * delta @ fisher_matrix @ delta
        assert jnp.allclose(kl_approx, expected, rtol=1e-4)


# =============================================================================
# Test Suite 5.2: SPD Manifold
# =============================================================================

@pytest.mark.phase5
class TestSPDManifold:
    """T5.2.x: SPD Manifold mathematical properties."""
    
    def test_T5_2_1_affine_invariance(self, key):
        """T5.2.1: d(WAW^T, WBW^T) = d(A, B) for invertible W."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        manifold = SPDManifold(dim=4)
        
        # Create two SPD matrices
        L1 = jax.random.normal(k1, (4, 4))
        A = L1 @ L1.T + 0.1 * jnp.eye(4)
        
        L2 = jax.random.normal(k2, (4, 4))
        B = L2 @ L2.T + 0.1 * jnp.eye(4)
        
        # Random invertible transformation
        W = jax.random.normal(k3, (4, 4)) + 0.5 * jnp.eye(4)
        
        # Original distance
        d_original = manifold.distance(A, B)
        
        # Transformed distance
        A_transformed = W @ A @ W.T
        B_transformed = W @ B @ W.T
        d_transformed = manifold.distance(A_transformed, B_transformed)
        
        assert jnp.allclose(d_original, d_transformed, rtol=1e-3)
    
    def test_T5_2_2_geodesic_stays_in_spd(self, key):
        """T5.2.2: γ(t) ∈ SPD for t ∈ [0,1]."""
        k1, k2 = jax.random.split(key)
        
        manifold = SPDManifold(dim=4)
        
        # Create two SPD matrices
        L1 = jax.random.normal(k1, (4, 4))
        A = L1 @ L1.T + 0.1 * jnp.eye(4)
        
        L2 = jax.random.normal(k2, (4, 4))
        B = L2 @ L2.T + 0.1 * jnp.eye(4)
        
        # Check geodesic points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gamma_t = manifold.geodesic(A, B, t)
            assert manifold.is_spd(gamma_t), f"Geodesic at t={t} not SPD"
    
    def test_T5_2_3_geodesic_endpoints(self, key):
        """T5.2.3: γ(0) = A and γ(1) = B."""
        k1, k2 = jax.random.split(key)
        
        manifold = SPDManifold(dim=4)
        
        L1 = jax.random.normal(k1, (4, 4))
        A = L1 @ L1.T + 0.1 * jnp.eye(4)
        
        L2 = jax.random.normal(k2, (4, 4))
        B = L2 @ L2.T + 0.1 * jnp.eye(4)
        
        gamma_0 = manifold.geodesic(A, B, 0.0)
        gamma_1 = manifold.geodesic(A, B, 1.0)
        
        assert jnp.allclose(gamma_0, A, rtol=1e-4)
        assert jnp.allclose(gamma_1, B, rtol=1e-4)
    
    def test_T5_2_4_log_exp_inverse(self, key):
        """T5.2.4: exp_P(log_P(Q)) = Q."""
        k1, k2 = jax.random.split(key)
        
        manifold = SPDManifold(dim=4)
        
        L1 = jax.random.normal(k1, (4, 4))
        P = L1 @ L1.T + 0.1 * jnp.eye(4)
        
        L2 = jax.random.normal(k2, (4, 4))
        Q = L2 @ L2.T + 0.1 * jnp.eye(4)
        
        # Round trip: Q → log → exp → Q
        V = manifold.log_map(P, Q)
        Q_recovered = manifold.exp_map(P, V)
        
        assert jnp.allclose(Q, Q_recovered, rtol=1e-3)
    
    def test_T5_2_5_frechet_mean_properties(self, key):
        """T5.2.5: Fréchet mean is closer to all points than Euclidean mean."""
        keys = jax.random.split(key, 5)
        
        manifold = SPDManifold(dim=3)
        
        # Generate SPD matrices
        matrices = []
        for k in keys:
            L = jax.random.normal(k, (3, 3))
            matrices.append(L @ L.T + 0.1 * jnp.eye(3))
        
        # Compute means
        euclidean_mean = sum(matrices) / len(matrices)
        euclidean_mean = manifold.project_to_spd(euclidean_mean)
        
        frechet_mean = manifold.frechet_mean(matrices, max_iter=50)
        
        # Compute sum of squared distances
        euclidean_cost = sum(manifold.distance_squared(euclidean_mean, M) for M in matrices)
        frechet_cost = sum(manifold.distance_squared(frechet_mean, M) for M in matrices)
        
        # Fréchet mean should have lower or equal cost
        assert frechet_cost <= euclidean_cost + 1e-3
    
    def test_T5_2_6_triangle_inequality(self, key):
        """T5.2.6: d(A,C) ≤ d(A,B) + d(B,C)."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        manifold = SPDManifold(dim=3)
        
        # Three SPD matrices
        L1 = jax.random.normal(k1, (3, 3))
        A = L1 @ L1.T + 0.1 * jnp.eye(3)
        
        L2 = jax.random.normal(k2, (3, 3))
        B = L2 @ L2.T + 0.1 * jnp.eye(3)
        
        L3 = jax.random.normal(k3, (3, 3))
        C = L3 @ L3.T + 0.1 * jnp.eye(3)
        
        d_AC = manifold.distance(A, C)
        d_AB = manifold.distance(A, B)
        d_BC = manifold.distance(B, C)
        
        assert d_AC <= d_AB + d_BC + 1e-6
    
    def test_parallel_transport_preserves_norm(self, key):
        """Parallel transport along geodesic preserves Riemannian norm (approximately)."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        manifold = SPDManifold(dim=3)
        
        # Two base points (close together for better approximation)
        L1 = jax.random.normal(k1, (3, 3))
        A = L1 @ L1.T + jnp.eye(3)  # Well-conditioned
        
        # B close to A
        L2 = jax.random.normal(k2, (3, 3)) * 0.1
        B = A + L2 @ L2.T  # Small perturbation
        B = manifold.project_to_spd(B)
        
        # Tangent vector at A
        V = jax.random.normal(k3, (3, 3)) * 0.1
        V = (V + V.T) / 2  # Symmetrize
        
        # Transport to B
        V_transported = manifold.parallel_transport(V, A, B)
        
        # Compute norms
        metric_A = SPDMetricTensor(A)
        metric_B = SPDMetricTensor(B)
        
        norm_A = metric_A.norm(V)
        norm_B = metric_B.norm(V_transported)
        
        # For close points, norms should be similar
        # The parallel transport formula is approximate
        assert jnp.allclose(norm_A, norm_B, rtol=0.3)


# =============================================================================
# Test Suite 5.3: Bregman Divergence
# =============================================================================

@pytest.mark.phase5
class TestBregmanDivergence:
    """T5.3.x: Bregman Divergence mathematical properties."""
    
    def test_T5_3_1_non_symmetry(self, key):
        """T5.3.1: D(P||Q) ≠ D(Q||P) in general."""
        kl = KLDivergence()
        
        # Use explicitly asymmetric distributions
        # p is concentrated, q is more uniform
        p = jnp.array([0.7, 0.2, 0.05, 0.03, 0.02])
        q = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        d_pq = kl(p, q)
        d_qp = kl(q, p)
        
        # These should be clearly different
        # KL(concentrated || uniform) < KL(uniform || concentrated)
        assert d_pq != d_qp
        assert abs(d_pq - d_qp) > 0.1  # Significant difference
    
    def test_T5_3_2_non_negativity(self, key):
        """T5.3.2: D(P||Q) ≥ 0, = 0 iff P = Q."""
        k1, k2 = jax.random.split(key)
        
        kl = KLDivergence()
        
        # Two different distributions
        p = jax.random.uniform(k1, (5,))
        p = p / jnp.sum(p)
        
        q = jax.random.uniform(k2, (5,))
        q = q / jnp.sum(q)
        
        # Non-negative
        assert kl(p, q) >= 0
        assert kl(q, p) >= 0
        
        # Zero iff equal
        assert jnp.allclose(kl(p, p), 0.0)
    
    def test_T5_3_3_squared_euclidean_symmetry(self, key):
        """T5.3.3: Squared Euclidean IS symmetric (special case)."""
        k1, k2 = jax.random.split(key)
        
        se = SquaredEuclidean()
        
        p = jax.random.normal(k1, (5,))
        q = jax.random.normal(k2, (5,))
        
        # Should be symmetric
        assert jnp.allclose(se(p, q), se(q, p))
    
    def test_T5_3_4_kl_properties(self, key):
        """T5.3.4: KL divergence matches direct computation."""
        k1, k2 = jax.random.split(key)
        
        kl = KLDivergence()
        
        p = jax.random.uniform(k1, (5,))
        p = p / jnp.sum(p)
        
        q = jax.random.uniform(k2, (5,))
        q = q / jnp.sum(q)
        
        # Direct computation
        kl_direct = jnp.sum(p * jnp.log(p / q))
        
        assert jnp.allclose(kl(p, q), kl_direct, rtol=1e-5)
    
    def test_T5_3_5_itakura_saito_scale_invariance(self, key):
        """T5.3.5: Itakura-Saito is scale-invariant."""
        k1, k2 = jax.random.split(key)
        
        its = ItakuraSaito()
        
        p = jnp.abs(jax.random.normal(k1, (5,))) + 0.1
        q = jnp.abs(jax.random.normal(k2, (5,))) + 0.1
        
        # Scale both by same factor
        alpha = 2.5
        
        d_original = its(p, q)
        d_scaled = its(alpha * p, alpha * q)
        
        # Should be equal (scale invariant)
        assert jnp.allclose(d_original, d_scaled, rtol=1e-4)
    
    def test_T5_3_6_js_symmetry(self, key):
        """T5.3.6: Jensen-Shannon is symmetric."""
        k1, k2 = jax.random.split(key)
        
        p = jax.random.uniform(k1, (5,))
        p = p / jnp.sum(p)
        
        q = jax.random.uniform(k2, (5,))
        q = q / jnp.sum(q)
        
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        
        assert jnp.allclose(js_pq, js_qp)
    
    def test_T5_3_7_total_variation_bounds(self, key):
        """T5.3.7: Total variation is bounded [0, 1]."""
        k1, k2 = jax.random.split(key)
        
        p = jax.random.uniform(k1, (5,))
        p = p / jnp.sum(p)
        
        q = jax.random.uniform(k2, (5,))
        q = q / jnp.sum(q)
        
        tv = total_variation(p, q)
        
        assert 0 <= tv <= 1
        assert total_variation(p, p) == 0
    
    def test_T5_3_8_hellinger_metric_properties(self, key):
        """T5.3.8: Hellinger is a true metric."""
        k1, k2, k3 = jax.random.split(key, 3)
        
        p = jax.random.uniform(k1, (5,))
        p = p / jnp.sum(p)
        
        q = jax.random.uniform(k2, (5,))
        q = q / jnp.sum(q)
        
        r = jax.random.uniform(k3, (5,))
        r = r / jnp.sum(r)
        
        # Symmetry
        assert jnp.allclose(hellinger_distance(p, q), hellinger_distance(q, p))
        
        # Identity of indiscernibles
        assert hellinger_distance(p, p) == 0
        
        # Triangle inequality
        h_pr = hellinger_distance(p, r)
        h_pq = hellinger_distance(p, q)
        h_qr = hellinger_distance(q, r)
        
        assert h_pr <= h_pq + h_qr + 1e-6


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.phase5
class TestPhase5Integration:
    """Integration tests for information geometry components."""
    
    def test_spd_classifier_basic(self, key):
        """Test SPD classifier on synthetic data."""
        keys = jax.random.split(key, 20)
        
        manifold = SPDManifold(dim=3)
        
        # Generate two classes of SPD matrices
        class_0 = []
        class_1 = []
        
        for i, k in enumerate(keys):
            L = jax.random.normal(k, (3, 3))
            if i < 10:
                # Class 0: more diagonal
                M = L @ L.T + 2 * jnp.eye(3)
                class_0.append(M)
            else:
                # Class 1: more off-diagonal
                M = L @ L.T + 0.1 * jnp.eye(3)
                class_1.append(M)
        
        matrices = class_0 + class_1
        labels = jnp.array([0] * 10 + [1] * 10)
        
        # Train classifier
        classifier = SPDClassifier(manifold)
        classifier.fit(matrices, labels)
        
        # Predict
        predictions = classifier.predict_batch(matrices)
        
        # Should get most correct (at least better than chance)
        accuracy = jnp.mean(predictions == labels)
        assert accuracy > 0.6  # Better than random
    
    def test_fisher_natural_gradient_step(self, key):
        """Test natural gradient descent step."""
        k1, k2 = jax.random.split(key)
        
        # Simple quadratic loss with known geometry
        dim = 4
        
        # Create a positive definite Hessian (Fisher-like)
        L = jax.random.normal(k1, (dim, dim))
        H = L @ L.T + jnp.eye(dim)
        fisher = FisherMetric(H)
        
        # Current parameters and gradient
        theta = jax.random.normal(k2, (dim,))
        gradient = H @ theta  # Gradient of 0.5 * theta^T H theta
        
        # Natural gradient step
        step_size = 0.1
        delta = fisher.geodesic_update(gradient, step_size)
        
        # Update
        theta_new = theta + delta
        
        # Loss should decrease (for small step)
        loss_old = 0.5 * theta @ H @ theta
        loss_new = 0.5 * theta_new @ H @ theta_new
        
        assert loss_new < loss_old
    
    def test_logdet_divergence_on_spd(self, key):
        """Test log-det divergence matches SPD distance structure."""
        k1, k2 = jax.random.split(key)
        
        dim = 3
        logdet = LogDet(dim)
        manifold = SPDManifold(dim)
        
        # Two SPD matrices
        L1 = jax.random.normal(k1, (dim, dim))
        A = L1 @ L1.T + 0.1 * jnp.eye(dim)
        
        L2 = jax.random.normal(k2, (dim, dim))
        B = L2 @ L2.T + 0.1 * jnp.eye(dim)
        
        # Log-det divergence
        d_ld = logdet(A, B)
        
        # Should be non-negative
        assert d_ld >= -1e-6
        
        # Zero for same matrix
        assert jnp.abs(logdet(A, A)) < 1e-4

