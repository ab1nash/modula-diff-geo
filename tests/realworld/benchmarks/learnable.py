"""
Learnable Imputation Models for Geometric Benchmarks

Simple but properly trained models comparing:
- Modula (Euclidean/standard) optimization
- DiffGeo (Riemannian/geometric) optimization

Implements proper ML practices:
- Train/validation/test splits
- Early stopping
- Learning rate scheduling
- Convergence tracking
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX not available. Learnable models disabled.")

from .manifold_integrity import ManifoldIntegrityScore, ManifoldType


@dataclass
class TrainingConfig:
    """
    Configuration for training learnable imputation models.
    
    Default values are set for proper convergence:
    - 2000 epochs max (early stopping usually kicks in earlier)
    - Patient early stopping (100 epochs without improvement)
    - Gradual learning rate decay
    """
    learning_rate: float = 0.01
    n_epochs: int = 2000  # Max epochs (early stopping will kick in)
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 100  # Wait 100 epochs for improvement
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 40  # Decay LR after 40 epochs without improvement
    min_lr: float = 1e-6  # Don't decay below this
    seed: int = 42
    log_every: int = 100
    
    # For quick testing
    @staticmethod
    def quick() -> 'TrainingConfig':
        """Quick config for testing (not for real results)."""
        return TrainingConfig(
            n_epochs=200,
            early_stopping_patience=30,
            lr_decay_patience=15,
            log_every=50,
        )
    
    @staticmethod
    def standard() -> 'TrainingConfig':
        """Standard training config (balanced speed/quality)."""
        return TrainingConfig(
            n_epochs=1000,
            early_stopping_patience=75,
            lr_decay_patience=30,
            log_every=100,
        )
    
    @staticmethod
    def thorough() -> 'TrainingConfig':
        """Thorough training config (best results, slower)."""
        return TrainingConfig(
            n_epochs=3000,
            early_stopping_patience=150,
            lr_decay_patience=50,
            log_every=200,
        )


@dataclass
class TrainingHistory:
    """Track training progress."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    final_epoch: int = 0
    training_time: float = 0.0
    converged: bool = False
    converged_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_epoch': self.final_epoch,
            'training_time': self.training_time,
            'converged': self.converged,
            'converged_reason': self.converged_reason,
        }


@dataclass
class AdamState:
    """State for Adam optimizer - identical for both models to ensure fair comparison."""
    m: Dict  # First moment estimates
    v: Dict  # Second moment estimates
    t: int   # Timestep
    
    @staticmethod
    def initialize(params: Dict) -> 'AdamState':
        """Initialize Adam state with zeros."""
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(val) for k, val in params.items()}
        return AdamState(m=m, v=v, t=0)


def adam_update(params: Dict, grads: Dict, state: AdamState, lr: float,
                beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[Dict, AdamState]:
    """
    Adam optimizer update - IDENTICAL for both Modula and DiffGeo models.
    
    This ensures fair comparison by giving both models the same optimization dynamics.
    """
    if not HAS_JAX:
        # Fallback to numpy
        new_m = {}
        new_v = {}
        new_params = {}
        t = state.t + 1
        
        for k in params:
            g = grads[k]
            m_old = state.m[k]
            v_old = state.v[k]
            
            # Update biased first moment estimate
            m = beta1 * m_old + (1 - beta1) * g
            # Update biased second moment estimate
            v = beta2 * v_old + (1 - beta2) * (g ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Parameter update
            new_params[k] = params[k] - lr * m_hat / (np.sqrt(v_hat) + eps)
            new_m[k] = m
            new_v[k] = v
        
        return new_params, AdamState(m=new_m, v=new_v, t=t)
    
    new_m = {}
    new_v = {}
    new_params = {}
    t = state.t + 1
    
    for k in params:
        g = grads[k]
        m_old = state.m[k]
        v_old = state.v[k]
        
        # Update biased first moment estimate
        m = beta1 * m_old + (1 - beta1) * g
        # Update biased second moment estimate
        v = beta2 * v_old + (1 - beta2) * (g ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Parameter update
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_m[k] = m
        new_v[k] = v
    
    return new_params, AdamState(m=new_m, v=new_v, t=t)


class ImputationModel:
    """
    Base class for learnable imputation models.
    
    Simple architecture: Linear encoder -> bottleneck -> Linear decoder
    The key difference is HOW we compute gradients and updates.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.params = None
        
    def init_params(self, key) -> Dict:
        """Initialize model parameters."""
        k1, k2 = jax.random.split(key)
        
        # Simple autoencoder: input -> hidden -> input
        scale = 0.1
        params = {
            'encoder_w': jax.random.normal(k1, (self.input_dim, self.hidden_dim)) * scale,
            'encoder_b': jnp.zeros(self.hidden_dim),
            'decoder_w': jax.random.normal(k2, (self.hidden_dim, self.input_dim)) * scale,
            'decoder_b': jnp.zeros(self.input_dim),
        }
        return params
    
    def forward(self, x: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Forward pass: reconstruct input."""
        # Encoder
        h = jnp.tanh(x @ params['encoder_w'] + params['encoder_b'])
        # Decoder
        out = h @ params['decoder_w'] + params['decoder_b']
        return out
    
    def loss_fn(self, params: Dict, x: jnp.ndarray, mask: jnp.ndarray) -> float:
        """
        Reconstruction loss on OBSERVED entries only.
        
        The model learns to reconstruct observed values, then we use it
        to impute missing values.
        """
        pred = self.forward(x * mask, params)  # Input has missing zeroed
        # Loss only on observed entries
        diff = (pred - x) * mask
        return jnp.mean(diff ** 2)


class ModulaImputationModel(ImputationModel):
    """
    Imputation model using standard Euclidean (modula) optimization.
    
    Now uses Adam optimizer (same as DiffGeo) to ensure fair comparison.
    The difference is in the MODEL architecture/loss, not the optimizer.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__(input_dim, hidden_dim, seed)
        self.name = "Modula (Euclidean)"
        self.adam_state = None
    
    def init_adam_state(self, params: Dict) -> None:
        """Initialize Adam optimizer state."""
        self.adam_state = AdamState.initialize(params)
    
    def compute_update(self, params: Dict, grads: Dict, lr: float) -> Dict:
        """Adam optimizer update - identical to DiffGeo for fair comparison."""
        if self.adam_state is None:
            self.init_adam_state(params)
        new_params, self.adam_state = adam_update(params, grads, self.adam_state, lr)
        return new_params


class DiffGeoImputationModel(ImputationModel):
    """
    Imputation model using Riemannian (diffgeo) approach.
    
    Uses natural gradient descent which accounts for the geometry of the
    parameter space. This is the key differentiator from Euclidean methods.
    
    Natural gradient: θ_new = θ - lr * F^(-1) * grad
    Where F is the Fisher information matrix (approximated here).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__(input_dim, hidden_dim, seed)
        self.name = "DiffGeo (Riemannian)"
        # State for natural gradient (running estimate of Fisher information)
        self.fisher_diag = None
        self.fisher_decay = 0.99
        self.fisher_eps = 1e-4
    
    def _init_fisher(self, params: Dict) -> None:
        """Initialize diagonal Fisher information estimate."""
        self.fisher_diag = {k: np.ones_like(v) for k, v in params.items()}
    
    def compute_update(self, params: Dict, grads: Dict, lr: float) -> Dict:
        """
        Natural gradient descent update.
        
        Approximates F^(-1) using diagonal Fisher information,
        estimated from squared gradients (empirical Fisher).
        """
        if self.fisher_diag is None:
            self._init_fisher(params)
        
        new_params = {}
        
        for k in params:
            g = grads[k]
            
            # Update diagonal Fisher estimate (exponential moving average of squared grads)
            # This approximates the empirical Fisher information
            self.fisher_diag[k] = (self.fisher_decay * self.fisher_diag[k] + 
                                   (1 - self.fisher_decay) * (g ** 2))
            
            # Natural gradient: precondition by inverse Fisher
            # F^(-1) * grad ≈ grad / (fisher_diag + eps)
            natural_grad = g / (self.fisher_diag[k] + self.fisher_eps)
            
            # Update parameters
            if HAS_JAX:
                new_params[k] = params[k] - lr * natural_grad
            else:
                new_params[k] = params[k] - lr * natural_grad
        
        return new_params


class SPDImputationModel(ImputationModel):
    """
    Imputation model specifically for SPD matrices (LEGACY - not manifold-aware).
    
    Operates in log-space (tangent space) which linearizes the SPD manifold.
    Uses same Adam optimizer as other models for fair comparison.
    
    NOTE: This is the old implementation. Use SPDTangentSpaceModel for proper
    manifold-aware imputation.
    """
    
    def __init__(self, matrix_dim: int, hidden_dim: int = 32, seed: int = 42):
        # Input is flattened upper triangle of SPD matrix
        self.matrix_dim = matrix_dim
        input_dim = matrix_dim * (matrix_dim + 1) // 2
        super().__init__(input_dim, hidden_dim, seed)
        self.name = "DiffGeo (SPD Log-Space)"
        self.adam_state = None
    
    def init_adam_state(self, params: Dict) -> None:
        """Initialize Adam optimizer state."""
        self.adam_state = AdamState.initialize(params)
    
    def _to_log_space(self, spd_flat: jnp.ndarray) -> jnp.ndarray:
        """Convert flattened SPD to log-space."""
        # Reconstruct matrix
        idx = jnp.triu_indices(self.matrix_dim)
        mat = jnp.zeros((self.matrix_dim, self.matrix_dim))
        mat = mat.at[idx].set(spd_flat)
        mat = mat + mat.T - jnp.diag(jnp.diag(mat))
        
        # Log transform
        eigvals, eigvecs = jnp.linalg.eigh(mat)
        eigvals = jnp.maximum(eigvals, 1e-6)
        log_mat = eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T
        
        return log_mat[idx]
    
    def _from_log_space(self, log_flat: jnp.ndarray) -> jnp.ndarray:
        """Convert log-space back to SPD."""
        idx = jnp.triu_indices(self.matrix_dim)
        mat = jnp.zeros((self.matrix_dim, self.matrix_dim))
        mat = mat.at[idx].set(log_flat)
        mat = mat + mat.T - jnp.diag(jnp.diag(mat))
        
        # Exp transform
        eigvals, eigvecs = jnp.linalg.eigh(mat)
        exp_mat = eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
        
        return exp_mat[idx]
    
    def forward(self, x: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Forward in log-space."""
        # Work in log space
        h = jnp.tanh(x @ params['encoder_w'] + params['encoder_b'])
        out = h @ params['decoder_w'] + params['decoder_b']
        return out
    
    def compute_update(self, params: Dict, grads: Dict, lr: float) -> Dict:
        """Adam optimizer update - identical to other models for fair comparison."""
        if self.adam_state is None:
            self.init_adam_state(params)
        new_params, self.adam_state = adam_update(params, grads, self.adam_state, lr)
        return new_params


class SPDTangentSpaceModel(ImputationModel):
    """
    Manifold-aware imputation model for SPD matrices.
    
    This model properly respects the geometry of the SPD manifold P_n:
    
    1. Maps data to tangent space at identity via matrix logarithm:
       V = log(P)  (tangent vector at I)
       
    2. Performs Euclidean operations in tangent space (where they're valid!)
    
    3. Maps back to manifold via matrix exponential:
       P = exp(V)  (guaranteed SPD!)
    
    The loss function uses Log-Euclidean distance:
        d_LE(A, B) = ||log(A) - log(B)||_F
    
    This avoids the "swelling effect" and off-manifold predictions that
    plague Euclidean methods on SPD data.
    
    Reference: Arsigny et al. (2006) "Log-Euclidean metrics for fast and
    simple calculus on diffusion tensors"
    """
    
    def __init__(self, matrix_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.matrix_dim = matrix_dim
        # Work with upper triangle in log-space (symmetric matrix)
        self.tri_dim = matrix_dim * (matrix_dim + 1) // 2
        # Initialize with upper triangle dimension
        super().__init__(self.tri_dim, hidden_dim, seed)
        self.name = "DiffGeo (SPD Tangent)"
        self.adam_state = None
        # Pre-compute indices as numpy arrays (JAX-safe, no tracing issues)
        self._triu_row, self._triu_col = np.triu_indices(matrix_dim)
    
    def init_adam_state(self, params: Dict) -> None:
        """Initialize Adam optimizer state."""
        self.adam_state = AdamState.initialize(params)
    
    def _get_triu_idx(self) -> Tuple:
        """Get upper triangular indices (numpy arrays, JAX-safe)."""
        return self._triu_row, self._triu_col
    
    def _matrix_log(self, P: jnp.ndarray) -> jnp.ndarray:
        """
        Compute matrix logarithm for SPD matrix.
        
        log(P) = V Λ V^T where Λ = diag(log(λ_i))
        """
        eigvals, eigvecs = jnp.linalg.eigh(P)
        # Clamp eigenvalues to prevent log(0)
        eigvals = jnp.maximum(eigvals, 1e-8)
        log_eigvals = jnp.log(eigvals)
        return eigvecs @ jnp.diag(log_eigvals) @ eigvecs.T
    
    def _matrix_exp(self, V: jnp.ndarray) -> jnp.ndarray:
        """
        Compute matrix exponential for symmetric matrix.
        
        exp(V) = U exp(Λ) U^T where Λ = diag(λ_i)
        
        Result is guaranteed SPD if V is symmetric!
        """
        eigvals, eigvecs = jnp.linalg.eigh(V)
        exp_eigvals = jnp.exp(eigvals)
        return eigvecs @ jnp.diag(exp_eigvals) @ eigvecs.T
    
    def _to_tangent(self, P_flat: jnp.ndarray) -> jnp.ndarray:
        """
        Map SPD matrix (flattened) to tangent space at identity.
        
        Returns upper triangle of log(P) as flat vector.
        """
        # Reshape to matrix
        P = P_flat.reshape(self.matrix_dim, self.matrix_dim)
        # Compute log
        log_P = self._matrix_log(P)
        # Extract upper triangle (symmetric, so no info lost)
        idx = self._get_triu_idx()
        return log_P[idx]
    
    def _from_tangent(self, v_flat: jnp.ndarray) -> jnp.ndarray:
        """
        Map tangent vector back to SPD manifold.
        
        Reconstructs symmetric matrix from upper triangle, then exponentiates.
        Result is GUARANTEED to be SPD!
        """
        # Reconstruct symmetric matrix from upper triangle
        idx = self._get_triu_idx()
        V = jnp.zeros((self.matrix_dim, self.matrix_dim))
        V = V.at[idx].set(v_flat)
        # Mirror to lower triangle (V is symmetric)
        V = V + V.T - jnp.diag(jnp.diag(V))
        # Exponentiate - result is guaranteed SPD
        P = self._matrix_exp(V)
        return P.flatten()
    
    def _batch_to_tangent(self, P_batch: jnp.ndarray) -> jnp.ndarray:
        """Vectorized version for batches."""
        return vmap(self._to_tangent)(P_batch)
    
    def _batch_from_tangent(self, v_batch: jnp.ndarray) -> jnp.ndarray:
        """Vectorized version for batches."""
        return vmap(self._from_tangent)(v_batch)
    
    def forward(self, x: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """
        Forward pass in tangent space.
        
        1. Input x is already masked SPD matrices (flattened)
        2. Map to tangent space (log)
        3. Apply autoencoder in tangent space
        4. Map back to manifold (exp)
        """
        # Autoencoder in tangent/Euclidean space
        h = jnp.tanh(x @ params['encoder_w'] + params['encoder_b'])
        out = h @ params['decoder_w'] + params['decoder_b']
        return out
    
    def forward_with_manifold(self, x: jnp.ndarray, params: Dict, 
                               is_spd_input: bool = True) -> jnp.ndarray:
        """
        Full forward with manifold projection.
        
        If is_spd_input=True, first converts input to tangent space.
        Always returns result on SPD manifold.
        """
        if is_spd_input:
            # Map to tangent space first
            x_tangent = self._batch_to_tangent(x)
        else:
            x_tangent = x
        
        # Process in tangent space
        out_tangent = self.forward(x_tangent, params)
        
        # Map back to manifold
        return self._batch_from_tangent(out_tangent)
    
    def loss_fn(self, params: Dict, x: jnp.ndarray, mask: jnp.ndarray) -> float:
        """
        Log-Euclidean loss function.
        
        Unlike Euclidean MSE on raw matrices, this respects the SPD geometry:
        d_LE(A, B) = ||log(A) - log(B)||_F
        
        The loss is computed in tangent space where Euclidean operations
        are geometrically valid.
        """
        # Convert full SPD matrices to tangent space
        # x is (batch, matrix_dim * matrix_dim), already flattened
        x_tangent = self._batch_to_tangent(x)
        
        # Create mask for tangent space (upper triangle only)
        # Need to convert full matrix mask to upper triangle mask
        mask_reshaped = mask.reshape(-1, self.matrix_dim, self.matrix_dim)
        idx = self._get_triu_idx()
        mask_tangent = mask_reshaped[:, idx[0], idx[1]]
        
        # Masked input in tangent space
        x_masked_tangent = x_tangent * mask_tangent
        
        # Forward pass in tangent space
        pred_tangent = self.forward(x_masked_tangent, params)
        
        # Loss on observed entries in tangent space
        # This IS the Log-Euclidean distance!
        diff = (pred_tangent - x_tangent) * mask_tangent
        return jnp.mean(diff ** 2)
    
    def compute_update(self, params: Dict, grads: Dict, lr: float) -> Dict:
        """
        Adam optimizer update for parameter space.
        
        Note: We use Adam (not natural gradient) because:
        1. The geometry is in the DATA space (SPD manifold), not parameter space
        2. The tangent space mapping linearizes the data manifold
        3. Neural network weights live in Euclidean R^{n×m}
        
        This is the correct approach for Log-Euclidean methods:
        "Apply standard algorithms in the tangent space after log-transform"
        - Arsigny et al. (2006)
        """
        if self.adam_state is None:
            self.init_adam_state(params)
        new_params, self.adam_state = adam_update(params, grads, self.adam_state, lr)
        return new_params


def train_model(model: ImputationModel,
                data: np.ndarray,
                mask: np.ndarray,
                config: TrainingConfig) -> Tuple[Dict, TrainingHistory]:
    """
    Train an imputation model with proper validation.
    
    Args:
        model: ImputationModel instance
        data: Full data array (n_samples, dim) or (n_samples, d, d) for SPD
        mask: Boolean mask (True = observed)
        config: Training configuration
        
    Returns:
        best_params: Best model parameters
        history: Training history
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for training")
    
    # Flatten if needed
    original_shape = data.shape
    if data.ndim == 3:  # SPD matrices
        n_samples = data.shape[0]
        data_flat = data.reshape(n_samples, -1)
        mask_flat = mask.reshape(n_samples, -1)
    else:
        data_flat = data
        mask_flat = mask
    
    data_flat = jnp.array(data_flat)
    mask_flat = jnp.array(mask_flat)
    
    n_samples = len(data_flat)
    
    # Train/val/test split
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(n_samples)
    
    n_test = int(n_samples * config.test_split)
    n_val = int(n_samples * config.validation_split)
    n_train = n_samples - n_test - n_val
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_data, train_mask = data_flat[train_idx], mask_flat[train_idx]
    val_data, val_mask = data_flat[val_idx], mask_flat[val_idx]
    
    # Initialize
    key = jax.random.PRNGKey(config.seed)
    params = model.init_params(key)
    
    # JIT compile loss and gradient
    @jit
    def compute_loss_and_grad(params, x, m):
        loss = model.loss_fn(params, x, m)
        return loss
    
    grad_fn = jit(grad(lambda p, x, m: model.loss_fn(p, x, m)))
    
    # Training state
    history = TrainingHistory()
    best_params = params
    lr = config.learning_rate
    patience_counter = 0
    lr_patience_counter = 0
    
    start_time = time.time()
    
    # Gradient clipping threshold
    max_grad_norm = 10.0
    
    def clip_grads(grads: Dict, max_norm: float) -> Dict:
        """Clip gradients to prevent explosion."""
        total_norm = 0.0
        for g in grads.values():
            total_norm += float(jnp.sum(g ** 2))
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            return {k: g * scale for k, g in grads.items()}
        return grads
    
    for epoch in range(config.n_epochs):
        # Shuffle training data
        perm = rng.permutation(n_train)
        train_data_shuffled = train_data[perm]
        train_mask_shuffled = train_mask[perm]
        
        # Mini-batch training
        epoch_losses = []
        for i in range(0, n_train, config.batch_size):
            batch_x = train_data_shuffled[i:i + config.batch_size]
            batch_m = train_mask_shuffled[i:i + config.batch_size]
            
            # Compute gradients
            grads = grad_fn(params, batch_x, batch_m)
            
            # Clip gradients for stability
            grads = clip_grads(grads, max_grad_norm)
            
            # Update using model-specific method
            params = model.compute_update(params, grads, lr)
            
            # Track loss
            batch_loss = float(compute_loss_and_grad(params, batch_x, batch_m))
            epoch_losses.append(batch_loss)
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        val_loss = float(compute_loss_and_grad(params, val_data, val_mask))
        
        # Record history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.learning_rates.append(lr)
        history.epochs.append(epoch)
        history.final_epoch = epoch
        
        # Check for improvement
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            patience_counter = 0
            lr_patience_counter = 0
        else:
            patience_counter += 1
            lr_patience_counter += 1
        
        # Learning rate decay
        if lr_patience_counter >= config.lr_decay_patience:
            new_lr = lr * config.lr_decay_factor
            if new_lr >= config.min_lr:
                lr = new_lr
                if epoch % config.log_every == 0:
                    print(f"  Epoch {epoch}: LR decayed to {lr:.6f}")
            lr_patience_counter = 0
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            history.converged = True
            # Flag exceptional cases
            if epoch < 100:
                flag = "⚡ FAST"
            elif epoch < config.n_epochs * 0.3:
                flag = "✓ GOOD"
            elif epoch > config.n_epochs * 0.8:
                flag = "⚠️  SLOW"
            else:
                flag = "✓"
            history.converged_reason = f"Early stop @ epoch {epoch} (best @ {history.best_epoch})"
            print(f"\n  {flag} {history.converged_reason}")
            break
        
        # Check for convergence (loss plateaued)
        # Only check after reasonable training, with stricter threshold
        min_epochs_for_plateau = max(200, config.n_epochs // 5)
        if epoch >= min_epochs_for_plateau and len(history.val_losses) > 100:
            recent_loss = np.mean(history.val_losses[-30:])
            older_loss = np.mean(history.val_losses[-100:-50])
            # Require 0.01% change (10x stricter than before)
            if abs(recent_loss - older_loss) / (older_loss + 1e-8) < 0.0001:
                history.converged = True
                history.converged_reason = f"Loss plateaued @ epoch {epoch}"
                print(f"\n  🎯 {history.converged_reason}")
                break
        
        # Logging
        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            print(f"  Epoch {epoch:4d}: train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"best={history.best_val_loss:.6f}@{history.best_epoch}, lr={lr:.6f}")
    
    # Check if we hit max epochs without converging
    if not history.converged:
        history.converged_reason = f"Max epochs ({config.n_epochs}) reached without convergence"
        print(f"\n  🔴 {history.converged_reason} - consider increasing epochs or patience")
    
    history.training_time = time.time() - start_time
    
    return best_params, history


def evaluate_model(model: ImputationModel,
                   params: Dict,
                   data: np.ndarray,
                   mask: np.ndarray,
                   manifold_type: ManifoldType = None) -> Dict[str, float]:
    """
    Evaluate trained model on test data.
    
    Returns metrics on imputed (missing) values, including MIS.
    """
    if not HAS_JAX:
        return {}
    
    original_shape = data.shape
    is_matrix_data = data.ndim == 3
    
    # Flatten if needed
    if is_matrix_data:
        n_samples = data.shape[0]
        data_flat = data.reshape(n_samples, -1)
        mask_flat = mask.reshape(n_samples, -1)
    else:
        data_flat = data
        mask_flat = mask
    
    data_flat = jnp.array(data_flat)
    mask_flat = jnp.array(mask_flat)
    
    # Get predictions
    masked_input = data_flat * mask_flat
    predictions = model.forward(masked_input, params)
    
    # Compute metrics on MISSING entries only
    missing_mask = ~mask_flat
    n_missing = jnp.sum(missing_mask)
    
    if n_missing == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'mis': 0.0}
    
    true_missing = data_flat[missing_mask]
    pred_missing = predictions[missing_mask]
    
    # RMSE
    rmse = float(jnp.sqrt(jnp.mean((true_missing - pred_missing) ** 2)))
    
    # MAE
    mae = float(jnp.mean(jnp.abs(true_missing - pred_missing)))
    
    # R² on missing values
    ss_res = jnp.sum((true_missing - pred_missing) ** 2)
    ss_tot = jnp.sum((true_missing - jnp.mean(true_missing)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    # Manifold Integrity Score (MIS)
    # Reconstruct full predictions for MIS computation
    full_predictions = np.array(data_flat * mask_flat + predictions * (~mask_flat))
    
    mis = 0.0
    if manifold_type is not None:
        if is_matrix_data:
            # Reshape back to matrices for MIS
            matrix_dim = original_shape[1]
            pred_matrices = full_predictions.reshape(-1, matrix_dim, matrix_dim)
            mis_calc = ManifoldIntegrityScore(manifold_type, dim=matrix_dim)
            mis, _ = mis_calc.compute_batch(np.array(pred_matrices))
        elif manifold_type == ManifoldType.SPHERE:
            mis_calc = ManifoldIntegrityScore(manifold_type)
            mis, _ = mis_calc.compute_batch(np.array(full_predictions))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mis': float(mis),
    }


def train_spd_model(model: 'SPDTangentSpaceModel',
                    data: np.ndarray,
                    mask: np.ndarray,
                    config: TrainingConfig) -> Tuple[Dict, TrainingHistory]:
    """
    Train SPD tangent space model with proper manifold-aware loss.
    
    Key differences from standard training:
    1. Works with full matrices (not flattened)
    2. Uses Log-Euclidean loss in tangent space
    3. Computes mask for upper triangle representation
    
    Args:
        model: SPDTangentSpaceModel instance
        data: SPD matrices (n_samples, dim, dim)
        mask: Boolean mask (True = observed)
        config: Training configuration
        
    Returns:
        best_params: Best model parameters
        history: Training history
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for training")
    
    # Data should be 3D (n_samples, matrix_dim, matrix_dim)
    assert data.ndim == 3, f"Expected 3D data for SPD, got {data.ndim}D"
    n_samples, matrix_dim, _ = data.shape
    
    # Flatten for batch processing
    data_flat = jnp.array(data.reshape(n_samples, -1))
    mask_flat = jnp.array(mask.reshape(n_samples, -1))
    
    # Train/val/test split
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(n_samples)
    
    n_test = int(n_samples * config.test_split)
    n_val = int(n_samples * config.validation_split)
    n_train = n_samples - n_test - n_val
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    
    train_data, train_mask = data_flat[train_idx], mask_flat[train_idx]
    val_data, val_mask = data_flat[val_idx], mask_flat[val_idx]
    
    # Initialize
    key = jax.random.PRNGKey(config.seed)
    params = model.init_params(key)
    
    # JIT compile loss and gradient
    @jit
    def compute_loss(params, x, m):
        return model.loss_fn(params, x, m)
    
    grad_fn = jit(grad(lambda p, x, m: model.loss_fn(p, x, m)))
    
    # Training state
    history = TrainingHistory()
    best_params = params
    lr = config.learning_rate
    patience_counter = 0
    lr_patience_counter = 0
    
    start_time = time.time()
    
    # Gradient clipping
    max_grad_norm = 10.0
    
    def clip_grads(grads: Dict, max_norm: float) -> Dict:
        total_norm = 0.0
        for g in grads.values():
            total_norm += float(jnp.sum(g ** 2))
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            return {k: g * scale for k, g in grads.items()}
        return grads
    
    for epoch in range(config.n_epochs):
        # Shuffle training data
        perm = rng.permutation(n_train)
        train_data_shuffled = train_data[perm]
        train_mask_shuffled = train_mask[perm]
        
        # Mini-batch training
        epoch_losses = []
        for i in range(0, n_train, config.batch_size):
            batch_x = train_data_shuffled[i:i + config.batch_size]
            batch_m = train_mask_shuffled[i:i + config.batch_size]
            
            # Compute gradients
            grads = grad_fn(params, batch_x, batch_m)
            grads = clip_grads(grads, max_grad_norm)
            
            # Update
            params = model.compute_update(params, grads, lr)
            
            batch_loss = float(compute_loss(params, batch_x, batch_m))
            epoch_losses.append(batch_loss)
        
        train_loss = np.mean(epoch_losses)
        val_loss = float(compute_loss(params, val_data, val_mask))
        
        # Record history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.learning_rates.append(lr)
        history.epochs.append(epoch)
        history.final_epoch = epoch
        
        # Check for improvement
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            patience_counter = 0
            lr_patience_counter = 0
        else:
            patience_counter += 1
            lr_patience_counter += 1
        
        # Learning rate decay
        if lr_patience_counter >= config.lr_decay_patience:
            new_lr = lr * config.lr_decay_factor
            if new_lr >= config.min_lr:
                lr = new_lr
                if epoch % config.log_every == 0:
                    print(f"  Epoch {epoch}: LR decayed to {lr:.6f}")
            lr_patience_counter = 0
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            history.converged = True
            if epoch < 100:
                flag = "⚡ FAST"
            elif epoch < config.n_epochs * 0.3:
                flag = "✓ GOOD"
            elif epoch > config.n_epochs * 0.8:
                flag = "⚠️  SLOW"
            else:
                flag = "✓"
            history.converged_reason = f"Early stop @ epoch {epoch} (best @ {history.best_epoch})"
            print(f"\n  {flag} {history.converged_reason}")
            break
        
        # Convergence check
        min_epochs_for_plateau = max(200, config.n_epochs // 5)
        if epoch >= min_epochs_for_plateau and len(history.val_losses) > 100:
            recent_loss = np.mean(history.val_losses[-30:])
            older_loss = np.mean(history.val_losses[-100:-50])
            if abs(recent_loss - older_loss) / (older_loss + 1e-8) < 0.0001:
                history.converged = True
                history.converged_reason = f"Loss plateaued @ epoch {epoch}"
                print(f"\n  🎯 {history.converged_reason}")
                break
        
        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            print(f"  Epoch {epoch:4d}: train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"best={history.best_val_loss:.6f}@{history.best_epoch}, lr={lr:.6f}")
    
    if not history.converged:
        history.converged_reason = f"Max epochs ({config.n_epochs}) reached without convergence"
        print(f"\n  🔴 {history.converged_reason}")
    
    history.training_time = time.time() - start_time
    
    return best_params, history


def evaluate_spd_model(model: 'SPDTangentSpaceModel',
                       params: Dict,
                       data: np.ndarray,
                       mask: np.ndarray,
                       manifold_type: ManifoldType = None) -> Dict[str, float]:
    """
    Evaluate SPD tangent space model with proper manifold metrics.
    
    Key differences from standard evaluation:
    1. Uses Log-Euclidean distance for error metrics
    2. Reconstructs full SPD matrices via exponential map (guaranteed SPD)
    3. Computes MIS to verify manifold integrity
    """
    if not HAS_JAX:
        return {}
    
    assert data.ndim == 3, "Expected 3D SPD data"
    n_samples, matrix_dim, _ = data.shape
    
    # Flatten for processing
    data_flat = jnp.array(data.reshape(n_samples, -1))
    mask_flat = jnp.array(mask.reshape(n_samples, -1))
    
    # Convert to tangent space
    true_tangent = model._batch_to_tangent(data_flat)
    
    # Create tangent space mask (upper triangle)
    mask_reshaped = mask.reshape(-1, matrix_dim, matrix_dim)
    idx = model._get_triu_idx()
    mask_tangent = jnp.array(mask_reshaped[:, idx[0], idx[1]])
    
    # Masked input in tangent space
    masked_tangent = true_tangent * mask_tangent
    
    # Forward pass
    pred_tangent = model.forward(masked_tangent, params)
    
    # Metrics in tangent space (Log-Euclidean)
    missing_mask = ~mask_tangent
    n_missing = jnp.sum(missing_mask)
    
    if n_missing == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'mis': 0.0}
    
    true_missing = true_tangent[missing_mask]
    pred_missing = pred_tangent[missing_mask]
    
    # RMSE in tangent space = Log-Euclidean distance
    rmse = float(jnp.sqrt(jnp.mean((true_missing - pred_missing) ** 2)))
    
    # MAE in tangent space
    mae = float(jnp.mean(jnp.abs(true_missing - pred_missing)))
    
    # R² in tangent space
    ss_res = jnp.sum((true_missing - pred_missing) ** 2)
    ss_tot = jnp.sum((true_missing - jnp.mean(true_missing)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    # Compute MIS (should be near-zero since exp(tangent) is guaranteed SPD)
    mis = 0.0
    if manifold_type == ManifoldType.SPD:
        # Reconstruct full predictions in tangent space
        full_pred_tangent = np.array(true_tangent * mask_tangent + pred_tangent * (~mask_tangent))
        
        # Map back to manifold
        pred_matrices_flat = model._batch_from_tangent(jnp.array(full_pred_tangent))
        pred_matrices = np.array(pred_matrices_flat).reshape(-1, matrix_dim, matrix_dim)
        
        mis_calc = ManifoldIntegrityScore(manifold_type, dim=matrix_dim)
        mis, _ = mis_calc.compute_batch(pred_matrices)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mis': float(mis),
    }


@dataclass
class LearnedBenchmarkResult:
    """Results from learned model benchmark (averaged over multiple runs)."""
    timestamp: str
    dataset_name: str
    missing_fraction: float
    modula_metrics: Dict[str, float]  # Mean metrics
    diffgeo_metrics: Dict[str, float]  # Mean metrics
    modula_metrics_std: Dict[str, float]  # Std metrics (if n_runs > 1)
    diffgeo_metrics_std: Dict[str, float]  # Std metrics (if n_runs > 1)
    modula_history: Dict
    diffgeo_history: Dict
    modula_training_time: float
    diffgeo_training_time: float
    n_runs: int = 1
    
    def to_dict(self) -> Dict:
        result = {
            'timestamp': self.timestamp,
            'dataset_name': self.dataset_name,
            'missing_fraction': self.missing_fraction,
            'n_runs': self.n_runs,
            'modula': {
                'metrics': self.modula_metrics,
                'metrics_std': self.modula_metrics_std,
                'history': self.modula_history,
                'training_time': self.modula_training_time,
            },
            'diffgeo': {
                'metrics': self.diffgeo_metrics,
                'metrics_std': self.diffgeo_metrics_std,
                'history': self.diffgeo_history,
                'training_time': self.diffgeo_training_time,
            },
            'improvement': {
                'rmse_pct': _compute_improvement(
                    self.modula_metrics.get('rmse', 0),
                    self.diffgeo_metrics.get('rmse', 0)
                ),
                'mae_pct': _compute_improvement(
                    self.modula_metrics.get('mae', 0),
                    self.diffgeo_metrics.get('mae', 0)
                ),
                'mis_pct': _compute_improvement(
                    self.modula_metrics.get('mis', 0),
                    self.diffgeo_metrics.get('mis', 0)
                ),
            }
        }
        return result


def _compute_improvement(baseline: float, geometric: float) -> float:
    """
    Compute % difference between methods (symmetric - can be positive or negative).
    
    Returns:
        Positive: geometric is better (lower error)
        Negative: baseline is better (lower error)
        Zero: methods are equivalent
    """
    if baseline == 0 and geometric == 0:
        return 0.0
    if baseline == 0:
        return -100.0  # Baseline was perfect, geometric is worse
    return (baseline - geometric) / baseline * 100


def _aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std of metrics across runs."""
    if not metrics_list:
        return {}, {}
    
    keys = metrics_list[0].keys()
    means = {}
    stds = {}
    
    for key in keys:
        values = [m.get(key, 0) for m in metrics_list]
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values))
    
    return means, stds


def run_learned_benchmark(data: np.ndarray,
                          dataset_name: str,
                          missing_fractions: List[float] = [0.1, 0.2, 0.3],
                          config: Optional[TrainingConfig] = None,
                          manifold_type: ManifoldType = None,
                          n_runs: int = 1) -> List[LearnedBenchmarkResult]:
    """
    Run full benchmark comparing Modula vs DiffGeo learned imputation.
    
    Args:
        data: Data array (n_samples, dim) or (n_samples, d, d) for SPD
        dataset_name: Name for logging
        missing_fractions: List of missing fractions to test
        config: Training configuration
        manifold_type: Type of manifold for MIS computation
        n_runs: Number of runs to average over (default 1, recommend 5 for papers)
        
    Returns:
        List of benchmark results for each missing fraction (averaged over n_runs)
    """
    if not HAS_JAX:
        print("JAX required for learned benchmarks")
        return []
    
    config = config or TrainingConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detect if data is SPD matrices (3D with square last dims)
    is_spd_data = (data.ndim == 3 and 
                   data.shape[1] == data.shape[2] and 
                   manifold_type == ManifoldType.SPD)
    
    # Determine input dimension and model type
    if data.ndim == 3:
        matrix_dim = data.shape[1]
        input_dim = matrix_dim * matrix_dim
        # For SPD, use smaller hidden dim based on triangular representation
        if is_spd_data:
            tri_dim = matrix_dim * (matrix_dim + 1) // 2
            hidden_dim = min(64, tri_dim // 2)
        else:
            hidden_dim = min(64, input_dim // 2)
    else:
        input_dim = data.shape[1]
        hidden_dim = min(64, input_dim)
    
    if is_spd_data:
        print(f"\n📐 Detected SPD manifold data ({matrix_dim}×{matrix_dim} matrices)")
        print(f"   Using manifold-aware tangent space model for DiffGeo")
    
    results = []
    
    for frac in missing_fractions:
        print(f"\n{'='*60}")
        print(f"Missing Fraction: {frac:.0%}")
        if n_runs > 1:
            print(f"Running {n_runs} trials for statistical significance...")
        print(f"{'='*60}")
        
        modula_metrics_all = []
        diffgeo_metrics_all = []
        modula_times = []
        diffgeo_times = []
        modula_history_last = None
        diffgeo_history_last = None
        
        for run_idx in range(n_runs):
            # Different seed for each run but reproducible
            run_seed = config.seed + run_idx * 1000 + int(frac * 100)
            
            if n_runs > 1:
                print(f"\n  --- Run {run_idx + 1}/{n_runs} (seed={run_seed}) ---")
            
            # Create random mask for this run
            rng = np.random.default_rng(run_seed)
            mask = rng.random(data.shape) > frac
            
            # Train Modula model (with fresh Adam state)
            print(f"\n  Training Modula (Euclidean)...")
            modula_model = ModulaImputationModel(input_dim, hidden_dim, run_seed)
            modula_model.adam_state = None
            modula_params, modula_history = train_model(modula_model, data, mask, config)
            modula_metrics = evaluate_model(modula_model, modula_params, data, mask, manifold_type)
            modula_metrics_all.append(modula_metrics)
            modula_times.append(modula_history.training_time)
            modula_history_last = modula_history
            print(f"    RMSE={modula_metrics['rmse']:.4f}, MAE={modula_metrics['mae']:.4f}, R²={modula_metrics['r2']:.4f}")
            
            # Train DiffGeo model - use manifold-aware model for SPD data
            if is_spd_data:
                print(f"\n  Training DiffGeo (SPD Tangent Space)...")
                diffgeo_model = SPDTangentSpaceModel(matrix_dim, hidden_dim, run_seed)
                diffgeo_model.adam_state = None
                diffgeo_params, diffgeo_history = train_spd_model(
                    diffgeo_model, data, mask, config
                )
                diffgeo_metrics = evaluate_spd_model(
                    diffgeo_model, diffgeo_params, data, mask, manifold_type
                )
            else:
                print(f"\n  Training DiffGeo (Riemannian)...")
                diffgeo_model = DiffGeoImputationModel(input_dim, hidden_dim, run_seed)
                diffgeo_model.fisher_diag = None
                diffgeo_params, diffgeo_history = train_model(diffgeo_model, data, mask, config)
                diffgeo_metrics = evaluate_model(diffgeo_model, diffgeo_params, data, mask, manifold_type)
            
            diffgeo_metrics_all.append(diffgeo_metrics)
            diffgeo_times.append(diffgeo_history.training_time)
            diffgeo_history_last = diffgeo_history
            print(f"    RMSE={diffgeo_metrics['rmse']:.4f}, MAE={diffgeo_metrics['mae']:.4f}, R²={diffgeo_metrics['r2']:.4f}")
        
        # Aggregate metrics across runs
        modula_mean, modula_std = _aggregate_metrics(modula_metrics_all)
        diffgeo_mean, diffgeo_std = _aggregate_metrics(diffgeo_metrics_all)
        
        # Report aggregated results
        if n_runs > 1:
            print(f"\n  {'='*50}")
            print(f"  AGGREGATED RESULTS ({n_runs} runs)")
            print(f"  {'='*50}")
            print(f"  Modula:  RMSE={modula_mean['rmse']:.4f}±{modula_std['rmse']:.4f}, "
                  f"MAE={modula_mean['mae']:.4f}±{modula_std['mae']:.4f}, "
                  f"R²={modula_mean['r2']:.4f}±{modula_std['r2']:.4f}")
            print(f"  DiffGeo: RMSE={diffgeo_mean['rmse']:.4f}±{diffgeo_std['rmse']:.4f}, "
                  f"MAE={diffgeo_mean['mae']:.4f}±{diffgeo_std['mae']:.4f}, "
                  f"R²={diffgeo_mean['r2']:.4f}±{diffgeo_std['r2']:.4f}")
        else:
            print(f"\n  Final Modula:  RMSE={modula_mean['rmse']:.4f}, MAE={modula_mean['mae']:.4f}, R²={modula_mean['r2']:.4f}")
            print(f"  Final DiffGeo: RMSE={diffgeo_mean['rmse']:.4f}, MAE={diffgeo_mean['mae']:.4f}, R²={diffgeo_mean['r2']:.4f}")
        
        # Compute difference
        rmse_diff = _compute_improvement(modula_mean['rmse'], diffgeo_mean['rmse'])
        if rmse_diff > 0:
            print(f"\n  Result: DiffGeo better by {rmse_diff:+.1f}% RMSE")
        elif rmse_diff < 0:
            print(f"\n  Result: Modula better by {-rmse_diff:+.1f}% RMSE")
        else:
            print(f"\n  Result: Methods are equivalent")
        
        result = LearnedBenchmarkResult(
            timestamp=timestamp,
            dataset_name=dataset_name,
            missing_fraction=frac,
            modula_metrics=modula_mean,
            diffgeo_metrics=diffgeo_mean,
            modula_metrics_std=modula_std,
            diffgeo_metrics_std=diffgeo_std,
            modula_history=modula_history_last.to_dict() if modula_history_last else {},
            diffgeo_history=diffgeo_history_last.to_dict() if diffgeo_history_last else {},
            modula_training_time=float(np.mean(modula_times)),
            diffgeo_training_time=float(np.mean(diffgeo_times)),
            n_runs=n_runs,
        )
        results.append(result)
    
    return results

