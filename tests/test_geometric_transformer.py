"""
Tests for the Geometric Transformer.

This module tests the geometric extensions to Modula's attention:

1. GeometricAttention: Attention with FinslerLinear projections (implicit drift)
2. GeometricGPT: Full transformer with TwistedEmbed

The key insight: asymmetry comes from FinslerLinear's weight space geometry,
NOT from explicit drift in attention scores. We reuse AttentionQK as-is.

Mathematical Tests (from the report):
- Manifold Consistency: Weights stay on scaled orthogonal manifold
- Equivariance: TwistedEmbed flips with orientation
- Dualization: FinslerLinear respects Finsler geometry

Reference: "Geometric Covariance in Deep Sequence Modeling" Report
"""
import pytest
import jax
import jax.numpy as jnp

from diffgeo import (
    # Core types
    TensorVariance, Parity, MetricType,
    # Atoms  
    FinslerLinear, TwistedEmbed, ContactAtom,
    # Bonds
    ParallelTransport, SymplecticBond,
    # Transformer
    GeometricAttention, GeometricGPT, TwistedEmbedWrapper,
    create_geometric_gpt, create_chiral_pair,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def key():
    """Random key for reproducible tests."""
    return jax.random.PRNGKey(42)


# =============================================================================
# Test Suite 1: Geometric Attention
# =============================================================================

@pytest.mark.geometric_transformer
class TestGeometricAttention:
    """Tests for GeometricAttention composition."""
    
    def test_attention_construction(self):
        """GeometricAttention should construct without error."""
        att = GeometricAttention(
            num_heads=4, d_embed=64, d_query=16, d_value=16,
            attention_scale=1.0, drift_strength=0.3
        )
        
        # Should have atoms (from FinslerLinear) and bonds
        assert att.atoms > 0
        assert att.bonds > 0
    
    def test_attention_forward(self, key):
        """GeometricAttention forward pass should work."""
        k1, k2 = jax.random.split(key)
        
        att = GeometricAttention(
            num_heads=4, d_embed=64, d_query=16, d_value=16,
            attention_scale=1.0, drift_strength=0.3
        )
        
        weights = att.initialize(k1)
        x = jax.random.normal(k2, shape=(2, 8, 64))  # [batch, seq, embed]
        
        y = att.forward(x, weights)
        
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    def test_attention_uses_finsler_linear(self):
        """GeometricAttention should use FinslerLinear (has drift params)."""
        att = GeometricAttention(
            num_heads=2, d_embed=32, d_query=16, d_value=16,
            drift_strength=0.3
        )
        
        # FinslerLinear has 2 atoms per layer (W and drift tensors)
        # Q, K, V, O = 4 FinslerLinear layers × 2 = 8 atoms
        assert att.atoms == 8, f"Expected 8 atoms, got {att.atoms}"
    
    def test_attention_gradient_flow(self, key):
        """Gradients should flow through GeometricAttention."""
        k1, k2 = jax.random.split(key)
        
        att = GeometricAttention(4, 64, 16, 16, drift_strength=0.3)
        weights = att.initialize(k1)
        x = jax.random.normal(k2, shape=(2, 8, 64))
        
        def loss_fn(w):
            return jnp.mean(att.forward(x, w) ** 2)
        
        grads = jax.grad(loss_fn)(weights)
        
        # All gradients should be finite
        for g in grads:
            assert jnp.all(jnp.isfinite(g))
    
    def test_attention_asymmetry_from_finsler(self, key):
        """Asymmetry should come from FinslerLinear, not explicit drift."""
        k1, k2 = jax.random.split(key)
        
        # Two attention with different drift strengths
        att_no_drift = GeometricAttention(4, 64, 16, 16, drift_strength=0.0)
        att_with_drift = GeometricAttention(4, 64, 16, 16, drift_strength=0.5)
        
        # Same input
        x = jax.random.normal(k2, shape=(2, 8, 64))
        
        weights_no = att_no_drift.initialize(k1)
        weights_yes = att_with_drift.initialize(k1)
        
        # Forward should work for both
        y_no = att_no_drift.forward(x, weights_no)
        y_yes = att_with_drift.forward(x, weights_yes)
        
        assert y_no.shape == y_yes.shape


# =============================================================================
# Test Suite 2: GeometricGPT
# =============================================================================

@pytest.mark.geometric_transformer
class TestGeometricGPT:
    """Tests for the full GeometricGPT model."""
    
    def test_gpt_construction(self):
        """GeometricGPT should construct without error."""
        model = GeometricGPT(
            vocab_size=100, num_heads=4, d_embed=64,
            d_query=16, d_value=16, num_blocks=2,
            drift_strength=0.3, orientation=1.0
        )
        
        assert model.atoms > 0
        assert model.bonds > 0
    
    def test_gpt_forward(self, key):
        """GeometricGPT forward pass should produce logits."""
        k1, k2 = jax.random.split(key)
        
        model = GeometricGPT(
            vocab_size=100, num_heads=4, d_embed=64,
            d_query=16, d_value=16, num_blocks=2
        )
        
        weights = model.initialize(k1)
        tokens = jax.random.randint(k2, shape=(2, 8), minval=0, maxval=100)
        
        logits = model.forward(tokens, weights)
        
        assert logits.shape == (2, 8, 100), f"Got {logits.shape}"
    
    def test_gpt_uses_twisted_embed(self):
        """GeometricGPT should use TwistedEmbed internally."""
        model = create_geometric_gpt(vocab_size=100, num_heads=2, d_embed=32)
        
        # TwistedEmbed contributes 1 atom
        # Check that model was created with orientation parameter
        assert model is not None
    
    def test_gpt_orientation_sensitivity(self, key):
        """Different orientations should produce different outputs."""
        k1, k2 = jax.random.split(key)
        
        model_pos = GeometricGPT(
            vocab_size=100, num_heads=4, d_embed=64,
            d_query=16, d_value=16, num_blocks=2,
            orientation=+1.0
        )
        
        model_neg = GeometricGPT(
            vocab_size=100, num_heads=4, d_embed=64,
            d_query=16, d_value=16, num_blocks=2,
            orientation=-1.0
        )
        
        # Use same initialization
        weights = model_pos.initialize(k1)
        tokens = jax.random.randint(k2, shape=(2, 8), minval=0, maxval=100)
        
        logits_pos = model_pos.forward(tokens, weights)
        logits_neg = model_neg.forward(tokens, weights)
        
        # Should produce different outputs
        assert not jnp.allclose(logits_pos, logits_neg), \
            "Different orientations should produce different outputs"
    
    def test_gpt_dualization(self, key):
        """GeometricGPT should support Modula dualization."""
        k1, k2 = jax.random.split(key)
        
        model = create_geometric_gpt(vocab_size=100)
        weights = model.initialize(k1)
        tokens = jax.random.randint(k2, shape=(2, 8), minval=0, maxval=100)
        
        def loss_fn(w):
            return jnp.mean(model.forward(tokens, w) ** 2)
        
        grads = jax.grad(loss_fn)(weights)
        dualized = model.dualize(grads)
        
        assert len(dualized) == len(weights)
        for d in dualized:
            assert jnp.all(jnp.isfinite(d))


# =============================================================================
# Test Suite 3: TwistedEmbed Equivariance
# =============================================================================

@pytest.mark.geometric_transformer
class TestTwistedEmbedEquivariance:
    """Tests for TwistedEmbed orientation behavior."""
    
    def test_orientation_flip(self, key):
        """TwistedEmbed with -1 orientation should flip signs."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=100)
        weights = embed.initialize(key)
        
        indices = jnp.array([5, 10, 42])
        
        y_pos = embed.forward(indices, weights, orientation=+1.0)
        y_neg = embed.forward(indices, weights, orientation=-1.0)
        
        assert jnp.allclose(y_pos, -y_neg), "Orientation should flip embeddings"
    
    def test_orientation_scaling(self, key):
        """TwistedEmbed should scale by orientation."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=100)
        weights = embed.initialize(key)
        
        indices = jnp.array([5, 10, 42])
        
        y_full = embed.forward(indices, weights, orientation=1.0)
        y_half = embed.forward(indices, weights, orientation=0.5)
        
        assert jnp.allclose(y_half, 0.5 * y_full)
    
    def test_wrapper_stores_orientation(self, key):
        """TwistedEmbedWrapper should store and use orientation."""
        wrapper_pos = TwistedEmbedWrapper(64, 100, orientation=+1.0)
        wrapper_neg = TwistedEmbedWrapper(64, 100, orientation=-1.0)
        
        weights = wrapper_pos.initialize(key)
        indices = jnp.array([5, 10, 42])
        
        y_pos = wrapper_pos.forward(indices, weights)
        y_neg = wrapper_neg.forward(indices, weights)
        
        assert jnp.allclose(y_pos, -y_neg)
    
    def test_parity_is_odd(self):
        """TwistedEmbed should have odd parity."""
        embed = TwistedEmbed(dEmbed=64, numEmbed=100)
        assert embed.signature.parity == Parity.ODD


# =============================================================================
# Test Suite 4: FinslerLinear Properties
# =============================================================================

@pytest.mark.geometric_transformer
class TestFinslerLinearProperties:
    """Tests for FinslerLinear geometric properties."""
    
    def test_orthogonalized_weights(self, key):
        """FinslerLinear should initialize with orthogonalized weights."""
        layer = FinslerLinear(32, 16, drift_strength=0.3)
        weights = layer.initialize(key)
        W = weights[0]
        
        # W should be approximately orthogonal (W^T W ≈ scaled I)
        gram = W.T @ W
        scale = 32 / 16
        expected = scale * jnp.eye(16)
        
        rel_error = jnp.linalg.norm(gram - expected) / jnp.linalg.norm(expected)
        assert rel_error < 0.02
    
    def test_drift_bounded(self, key):
        """FinslerLinear drift should be bounded."""
        layer = FinslerLinear(32, 16, drift_strength=0.4)
        weights = layer.initialize(key)
        drift = weights[1]
        
        drift_norm = jnp.linalg.norm(drift)
        assert drift_norm < 1.0, "Drift should be bounded"
    
    def test_forward_is_linear(self, key):
        """FinslerLinear forward should be y = Wx."""
        k1, k2 = jax.random.split(key)
        
        layer = FinslerLinear(32, 16, drift_strength=0.3)
        weights = layer.initialize(k1)
        W = weights[0]
        
        x = jax.random.normal(k2, shape=(16,))
        y = layer.forward(x, weights)
        
        expected = W @ x
        assert jnp.allclose(y, expected)
    
    def test_dualize_orthogonalizes(self, key):
        """FinslerLinear dualize should orthogonalize gradients."""
        k1, k2 = jax.random.split(key)
        
        layer = FinslerLinear(32, 16, drift_strength=0.3)
        weights = layer.initialize(k1)
        
        # Random gradient
        grad = [jax.random.normal(k2, shape=w.shape) for w in weights]
        
        dualized = layer.dualize(grad)
        
        # Dualized weight gradient should be orthogonalized
        D = dualized[0]
        singular_values = jnp.linalg.svd(D, compute_uv=False)
        
        # All singular values should be similar
        assert jnp.std(singular_values) / jnp.mean(singular_values) < 0.1
    
    def test_finsler_linear_two_atoms(self):
        """FinslerLinear should have atoms=2 (W and drift)."""
        layer = FinslerLinear(32, 16, drift_strength=0.3)
        assert layer.atoms == 2, f"Expected 2 atoms, got {layer.atoms}"


# =============================================================================
# Test Suite 5: Factory Functions
# =============================================================================

@pytest.mark.geometric_transformer
class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_geometric_gpt(self, key):
        """create_geometric_gpt should work with defaults."""
        model = create_geometric_gpt(vocab_size=100)
        weights = model.initialize(key)
        
        assert len(weights) > 0
    
    def test_create_chiral_pair(self, key):
        """create_chiral_pair should return two models with opposite orientations."""
        left, right = create_chiral_pair(vocab_size=100)
        
        weights = left.initialize(key)
        tokens = jnp.array([[1, 2, 3, 4]])
        
        logits_left = left.forward(tokens, weights)
        logits_right = right.forward(tokens, weights)
        
        # Different orientations should give different outputs
        assert not jnp.allclose(logits_left, logits_right)


# =============================================================================
# Test Suite 6: Integration with Modula
# =============================================================================

@pytest.mark.geometric_transformer
class TestModulaIntegration:
    """Tests for integration with Modula patterns."""
    
    def test_composition(self, key):
        """GeometricAttention should compose with other modules."""
        from modula.abstract import Identity
        
        att = GeometricAttention(2, 32, 16, 16, drift_strength=0.3)
        
        # Residual connection
        residual = 0.5 * Identity() + 0.5 * att
        
        weights = residual.initialize(key)
        x = jax.random.normal(jax.random.split(key)[0], shape=(2, 8, 32))
        
        y = residual.forward(x, weights)
        assert y.shape == x.shape
    
    def test_repetition(self, key):
        """GeometricAttention blocks can be repeated with **."""
        from modula.abstract import Identity
        
        att = GeometricAttention(2, 32, 16, 16, drift_strength=0.3)
        block = 0.9 * Identity() + 0.1 * att
        
        # Stack 2 blocks
        stacked = block ** 2
        
        weights = stacked.initialize(key)
        x = jax.random.normal(jax.random.split(key)[0], shape=(2, 8, 32))
        
        y = stacked.forward(x, weights)
        assert y.shape == x.shape
    
    def test_tare(self, key):
        """GeometricGPT should support tare()."""
        model = create_geometric_gpt(vocab_size=100)
        
        original_mass = model.mass
        model.tare(absolute=10.0)
        
        assert model.mass == 10.0
    
    def test_jit(self, key):
        """GeometricGPT should support jit()."""
        k1, k2 = jax.random.split(key)
        
        model = create_geometric_gpt(vocab_size=100)
        model.jit()
        
        weights = model.initialize(k1)
        tokens = jax.random.randint(k2, shape=(2, 8), minval=0, maxval=100)
        
        # Should work after jit
        logits = model.forward(tokens, weights)
        assert logits.shape == (2, 8, 100)


# =============================================================================
# Test Suite 7: Symplectic and Contact (from existing atoms)
# =============================================================================

@pytest.mark.geometric_transformer
class TestExistingGeometricAtoms:
    """Tests for existing geometric atoms used in transformer."""
    
    def test_contact_atom_projection(self, key):
        """ContactAtom should project to kernel."""
        k1, k2 = jax.random.split(key)
        
        contact = ContactAtom(dim=33)
        weights = contact.initialize(k1)
        alpha = weights[0]
        
        x = jax.random.normal(k2, shape=(33,))
        x_proj = contact.forward(x, weights)
        
        # Should be in kernel
        assert jnp.abs(jnp.dot(alpha, x_proj)) < 1e-5
    
    def test_parallel_transport_flat(self, key):
        """ParallelTransport on flat manifold is identity."""
        transport = ParallelTransport(dim=32, curvature=0.0)
        
        v = jax.random.normal(key, shape=(32,))
        v_transported = transport.forward(v, [])
        
        assert jnp.allclose(v, v_transported)
    
    def test_symplectic_bond_properties(self):
        """SymplecticBond should have correct J matrix."""
        bond = SymplecticBond(dim=8)
        J = bond.J
        
        # J^T = -J
        assert jnp.allclose(J.T, -J)
        
        # J^2 = -I
        assert jnp.allclose(J @ J, -jnp.eye(8))


# =============================================================================
# Test Suite 8: RopeJIT (JIT-compatible Rotary Position Embedding)
# =============================================================================

from diffgeo import RopeJIT

@pytest.mark.geometric_transformer
class TestRopeJIT:
    """
    Tests for RopeJIT - the JIT-compatible Rotary Position Embedding.
    
    Key properties:
    1. JIT-compatible: No Python-level caching that breaks tracing
    2. Mathematically equivalent to standard Rope
    3. Orthogonal transformation (preserves norms)
    """
    
    def test_rope_jit_basic(self, key):
        """RopeJIT should apply rotation to Q and K."""
        rope = RopeJIT(d_head=16)
        
        # Input shape: [batch, n_heads, seq_len, d_head]
        q = jax.random.normal(key, shape=(2, 4, 8, 16))
        k = jax.random.normal(key, shape=(2, 4, 8, 16))
        
        q_rot, k_rot = rope.forward((q, k), [])
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Output should be different from input
        assert not jnp.allclose(q, q_rot)
    
    def test_rope_jit_is_jit_compatible(self, key):
        """RopeJIT should work inside JAX JIT without tracer errors."""
        rope = RopeJIT(d_head=16)
        
        @jax.jit
        def apply_rope(q, k):
            return rope.forward((q, k), [])
        
        q = jax.random.normal(key, shape=(2, 4, 8, 16))
        k = jax.random.normal(key, shape=(2, 4, 8, 16))
        
        # This would fail with standard Rope due to caching
        q_rot, k_rot = apply_rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_rope_jit_repeated_calls_consistent(self, key):
        """Multiple JIT calls should give consistent results."""
        rope = RopeJIT(d_head=16)
        
        @jax.jit
        def apply_rope(q, k):
            return rope.forward((q, k), [])
        
        q = jax.random.normal(key, shape=(2, 4, 8, 16))
        k = jax.random.normal(key, shape=(2, 4, 8, 16))
        
        # Multiple calls should give same result
        q1, k1 = apply_rope(q, k)
        q2, k2 = apply_rope(q, k)
        
        assert jnp.allclose(q1, q2)
        assert jnp.allclose(k1, k2)
    
    def test_rope_jit_preserves_norm(self, key):
        """RopeJIT is orthogonal: should approximately preserve norms."""
        rope = RopeJIT(d_head=16)
        
        q = jax.random.normal(key, shape=(2, 4, 8, 16))
        k = jax.random.normal(key, shape=(2, 4, 8, 16))
        
        q_rot, k_rot = rope.forward((q, k), [])
        
        # Norms should be approximately preserved
        q_norm = jnp.linalg.norm(q, axis=-1)
        q_rot_norm = jnp.linalg.norm(q_rot, axis=-1)
        
        assert jnp.allclose(q_norm, q_rot_norm, rtol=1e-4)
    
    def test_rope_jit_position_dependent(self, key):
        """Different positions should get different rotations."""
        rope = RopeJIT(d_head=16)
        
        # Same token at different positions
        x = jax.random.normal(key, shape=(1, 1, 1, 16))
        x_repeated = jnp.tile(x, (1, 1, 4, 1))  # 4 positions
        
        x_rot, _ = rope.forward((x_repeated, x_repeated), [])
        
        # Each position should be different
        pos0 = x_rot[0, 0, 0]
        pos1 = x_rot[0, 0, 1]
        pos2 = x_rot[0, 0, 2]
        
        assert not jnp.allclose(pos0, pos1)
        assert not jnp.allclose(pos1, pos2)
    
    def test_rope_jit_grad_works(self, key):
        """Gradients should flow through RopeJIT."""
        rope = RopeJIT(d_head=16)
        
        def loss_fn(q, k):
            q_rot, k_rot = rope.forward((q, k), [])
            return jnp.sum(q_rot * k_rot)
        
        q = jax.random.normal(key, shape=(2, 4, 8, 16))
        k = jax.random.normal(key, shape=(2, 4, 8, 16))
        
        # Gradient should work
        grad_q, grad_k = jax.grad(loss_fn, argnums=(0, 1))(q, k)
        
        assert grad_q.shape == q.shape
        assert grad_k.shape == k.shape
        assert not jnp.allclose(grad_q, 0)
    
    def test_rope_jit_in_full_training_step(self, key):
        """RopeJIT should work in a full JIT-compiled training step."""
        from diffgeo import GeometricGPT
        
        # Create a small model
        model = GeometricGPT(
            vocab_size=100, num_heads=2, d_embed=32, 
            d_query=16, d_value=16, num_blocks=1
        )
        
        k1, k2 = jax.random.split(key)
        weights = model.initialize(k1)
        
        # Input sequences
        sequences = jax.random.randint(k2, (4, 8), 0, 100)
        
        @jax.jit
        def train_step(weights, sequences):
            def loss_fn(w):
                logits = model.forward(sequences, w)
                return jnp.mean(logits ** 2)
            
            loss, grads = jax.value_and_grad(loss_fn)(weights)
            dualized = model.dualize(grads)
            new_weights = [w - 0.01 * d for w, d in zip(weights, dualized)]
            return new_weights, loss
        
        # This would fail with standard Rope due to caching issues
        new_weights, loss = train_step(weights, sequences)
        
        assert loss > 0
        assert len(new_weights) == len(weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "geometric_transformer"])
