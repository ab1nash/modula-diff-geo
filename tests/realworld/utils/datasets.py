"""
Synthetic Dataset Generators for Geometric Covariance Testing

Generate synthetic datasets mimicking real-world geometric structure,
designed to isolate and test specific geometric properties.
"""
from typing import Tuple

import jax
import jax.numpy as jnp


class SyntheticDatasets:
    """
    Generate synthetic datasets mimicking real-world geometric structure.
    
    These are simplified versions of the real datasets mentioned in the
    research document, designed to isolate and test specific geometric
    properties.
    """
    
    @staticmethod
    def generate_spd_matrices(
        n_samples: int,
        dim: int,
        n_classes: int,
        key: jax.Array,
        class_separation: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate synthetic SPD covariance matrices mimicking EEG data.
        
        From Section 4 of the doc:
        "In interpreting EEG signals, the spatial covariance matrix of
        electrode voltages captures the synchronization of brain regions."
        
        Each class has a distinct "brain state" characterized by different
        covariance structure (e.g., high correlation in certain regions).
        
        Returns:
            matrices: (n_samples, dim, dim) SPD matrices
            labels: (n_samples,) class labels
        """
        keys = jax.random.split(key, n_samples + n_classes)
        
        # Generate class-specific base covariance structures
        class_bases = []
        for i in range(n_classes):
            # Each class has distinct eigenstructure
            L = jax.random.normal(keys[i], (dim, dim))
            # Add class-specific bias to create separation
            bias = jnp.eye(dim) * (1 + i * class_separation)
            base = L @ L.T + bias
            class_bases.append(base)
        
        matrices = []
        labels = []
        for i in range(n_samples):
            class_idx = i % n_classes
            base = class_bases[class_idx]
            
            # Add noise while preserving SPD property
            noise = jax.random.normal(keys[n_classes + i], (dim, dim))
            noise = noise @ noise.T * 0.1
            
            matrix = base + noise
            matrices.append(matrix)
            labels.append(class_idx)
        
        return jnp.stack(matrices), jnp.array(labels)
    
    @staticmethod
    def generate_directed_graph_data(
        n_samples: int,
        n_nodes: int,
        key: jax.Array,
        asymmetry_strength: float = 0.5
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate directed graph adjacency matrices and node features.
        
        From Section 5 of the doc on GDL:
        "In drug discovery, molecules are treated as graphs."
        
        Models asymmetric relationships like:
        - Social influence (A influences B, but not vice versa)
        - Causal dependencies
        - Information flow
        
        Returns:
            adjacency: (n_samples, n_nodes, n_nodes) directed adjacency
            features: (n_samples, n_nodes, feature_dim) node features
            labels: (n_samples,) graph-level labels
        """
        keys = jax.random.split(key, n_samples * 3)
        
        adjacency_list = []
        features_list = []
        labels = []
        
        for i in range(n_samples):
            # Generate base symmetric structure
            k1, k2, k3 = keys[i*3:(i+1)*3]
            sym = jax.random.uniform(k1, (n_nodes, n_nodes))
            sym = (sym + sym.T) / 2  # Symmetric base
            
            # Add directed component (asymmetry)
            asym = jax.random.normal(k2, (n_nodes, n_nodes)) * asymmetry_strength
            asym = jnp.triu(asym) - jnp.triu(asym).T  # Antisymmetric
            
            adj = sym + asym
            adj = (adj > 0.5).astype(jnp.float32)  # Threshold to binary
            adj = adj - jnp.diag(jnp.diag(adj))  # No self-loops
            
            # Node features
            feats = jax.random.normal(k3, (n_nodes, 8))
            
            # Label based on graph property (e.g., net flow direction)
            flow = jnp.sum(jnp.triu(adj)) - jnp.sum(jnp.tril(adj))
            label = (flow > 0).astype(jnp.int32)
            
            adjacency_list.append(adj)
            features_list.append(feats)
            labels.append(label)
        
        return (jnp.stack(adjacency_list), 
                jnp.stack(features_list), 
                jnp.array(labels))
    
    @staticmethod
    def generate_chiral_data(
        n_samples: int,
        dim: int,
        key: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate pairs of chiral (mirror-image) data points.
        
        From Section 6 of the doc on biological vision:
        "Chiral discrimination" - distinguishing left from right-handed forms.
        
        Applications:
        - Drug molecules (L vs D amino acids)
        - Handwriting (left vs right slant)
        - Spiral galaxies
        
        Returns:
            data: (n_samples, dim) feature vectors
            chirality: (n_samples,) +1 for right-handed, -1 for left-handed
        """
        keys = jax.random.split(key, n_samples)
        
        data = []
        chirality = []
        
        for i in range(n_samples):
            # Generate a "handed" pattern
            base = jax.random.normal(keys[i], (dim,))
            
            # Create chirality by asymmetric component
            # (like a spiral that goes clockwise vs counterclockwise)
            hand = 1 if i % 2 == 0 else -1
            
            # Add chiral signature: cross-term asymmetry
            if dim >= 2:
                chiral_component = jnp.zeros(dim)
                chiral_component = chiral_component.at[0].set(hand * base[1])
                chiral_component = chiral_component.at[1].set(-hand * base[0])
                point = base + 0.3 * chiral_component
            else:
                point = base * hand
            
            data.append(point)
            chirality.append(hand)
        
        return jnp.stack(data), jnp.array(chirality)
    
    @staticmethod
    def generate_affine_transformed_data(
        n_samples: int,
        base_pattern: jnp.ndarray,
        key: jax.Array,
        transform_strength: float = 0.5
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate data with random affine transformations.
        
        From Section 6.1 of the doc:
        "Affine Covariance: When an object is viewed from a slant, its image
        undergoes an affine transformation (foreshortening)."
        
        Tests robustness to perspective distortion (viewing angle changes).
        
        Returns:
            transformed: (n_samples, dim) transformed patterns
            transforms: (n_samples, dim, dim) applied transformations
        """
        dim = base_pattern.shape[0]
        keys = jax.random.split(key, n_samples)
        
        transformed = []
        transforms = []
        
        for i in range(n_samples):
            # Random affine transformation (rotation + scaling + shear)
            k1, k2 = jax.random.split(keys[i])
            
            # Random matrix near identity
            A = jnp.eye(dim) + jax.random.normal(k1, (dim, dim)) * transform_strength
            
            # Ensure invertible (add small identity component)
            A = A + 0.1 * jnp.eye(dim)
            
            # Apply transformation
            point = A @ base_pattern
            
            transformed.append(point)
            transforms.append(A)
        
        return jnp.stack(transformed), jnp.stack(transforms)
    
    # =========================================================================
    # EUCLIDEAN-FAVORABLE SCENARIOS
    # These datasets are designed to test scenarios where Euclidean methods
    # should perform equally well or better than geometric methods.
    # =========================================================================
    
    @staticmethod
    def generate_low_rank_data(
        n_samples: int,
        dim: int,
        rank: int,
        key: jax.Array,
        noise_level: float = 0.01
    ) -> jnp.ndarray:
        """
        Generate low-rank data that lies near a linear subspace.
        
        This is a EUCLIDEAN-FAVORABLE scenario where:
        - Data lies on/near a flat (linear) manifold
        - SVD-based methods should excel
        - Geometric methods have no advantage
        
        Use this to verify that geometric methods don't artificially outperform
        when the data has no inherent geometric structure.
        
        Args:
            n_samples: Number of samples
            dim: Ambient dimension
            rank: Intrinsic rank of the data (rank << dim)
            key: JAX random key
            noise_level: Gaussian noise standard deviation
            
        Returns:
            data: (n_samples, dim) data matrix with approximate rank `rank`
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Generate low-rank factors
        U = jax.random.normal(k1, (n_samples, rank))
        V = jax.random.normal(k2, (rank, dim))
        
        # Low-rank data
        data = U @ V
        
        # Add small Gaussian noise
        noise = jax.random.normal(k3, (n_samples, dim)) * noise_level
        data = data + noise
        
        return data
    
    @staticmethod
    def generate_gaussian_data(
        n_samples: int,
        dim: int,
        key: jax.Array,
        mean: float = 0.0,
        std: float = 1.0
    ) -> jnp.ndarray:
        """
        Generate IID Gaussian data with no geometric structure.
        
        This is a EUCLIDEAN-FAVORABLE scenario where:
        - Data is isotropic Gaussian
        - No manifold structure exists
        - All methods should perform similarly
        - Geometric methods should NOT have an advantage
        
        Use this as a baseline to verify fair comparison.
        
        Args:
            n_samples: Number of samples
            dim: Dimension
            key: JAX random key
            mean: Mean of Gaussian
            std: Standard deviation
            
        Returns:
            data: (n_samples, dim) IID Gaussian samples
        """
        data = jax.random.normal(key, (n_samples, dim)) * std + mean
        return data
    
    @staticmethod
    def generate_near_identity_spd(
        n_samples: int,
        dim: int,
        key: jax.Array,
        perturbation_scale: float = 0.01
    ) -> jnp.ndarray:
        """
        Generate SPD matrices close to identity (low curvature region).
        
        This is a EUCLIDEAN-FAVORABLE scenario where:
        - Matrices are near the identity (flat region of SPD manifold)
        - Riemannian curvature effects are minimal
        - Euclidean and Riemannian metrics approximately coincide
        - Both methods should perform similarly
        
        At the identity matrix I, the SPD manifold is locally flat,
        so Euclidean operations are valid approximations.
        
        Args:
            n_samples: Number of SPD matrices
            dim: Matrix dimension
            key: JAX random key
            perturbation_scale: How far from identity (smaller = flatter)
            
        Returns:
            matrices: (n_samples, dim, dim) SPD matrices near identity
        """
        keys = jax.random.split(key, n_samples)
        matrices = []
        
        for i in range(n_samples):
            # Small symmetric perturbation
            noise = jax.random.normal(keys[i], (dim, dim)) * perturbation_scale
            noise = (noise + noise.T) / 2  # Symmetrize
            
            # I + small perturbation is SPD if perturbation is small enough
            # Add small diagonal to ensure positive definiteness
            matrix = jnp.eye(dim) + noise + 0.1 * perturbation_scale * jnp.eye(dim)
            matrices.append(matrix)
        
        return jnp.stack(matrices)
    
    @staticmethod
    def generate_linear_time_series(
        n_samples: int,
        seq_length: int,
        dim: int,
        key: jax.Array,
        trend_strength: float = 0.1
    ) -> jnp.ndarray:
        """
        Generate linear time series data.
        
        This is a EUCLIDEAN-FAVORABLE scenario where:
        - Data follows linear trends
        - Linear interpolation is optimal
        - Geodesic interpolation has no advantage
        
        Args:
            n_samples: Number of sequences
            seq_length: Length of each sequence
            dim: Feature dimension
            key: JAX random key
            trend_strength: Magnitude of linear trend
            
        Returns:
            data: (n_samples, seq_length, dim) linear time series
        """
        k1, k2 = jax.random.split(key)
        
        # Time points
        t = jnp.linspace(0, 1, seq_length)
        
        # Random starting points and slopes
        starts = jax.random.normal(k1, (n_samples, dim))
        slopes = jax.random.normal(k2, (n_samples, dim)) * trend_strength
        
        # Generate linear sequences
        data = []
        for i in range(n_samples):
            seq = starts[i] + jnp.outer(t, slopes[i])
            data.append(seq)
        
        return jnp.stack(data)
    
    @staticmethod
    def generate_spherical_near_pole(
        n_samples: int,
        key: jax.Array,
        spread: float = 0.1
    ) -> jnp.ndarray:
        """
        Generate spherical data concentrated near a pole.
        
        This is a EUCLIDEAN-FAVORABLE scenario where:
        - Data is concentrated in a small region of the sphere
        - Local geometry is approximately flat
        - Euclidean operations are good approximations
        
        Near a pole, the sphere is locally flat, so great circle distances
        approximately equal Euclidean distances.
        
        Args:
            n_samples: Number of points
            key: JAX random key
            spread: Angular spread around pole (radians)
            
        Returns:
            data: (n_samples, 2) lat/lon coordinates near north pole
        """
        k1, k2 = jax.random.split(key)
        
        # Latitude near pi/2 (north pole)
        lat = jnp.pi / 2 - jnp.abs(jax.random.normal(k1, (n_samples,))) * spread
        lat = jnp.clip(lat, -jnp.pi/2, jnp.pi/2)
        
        # Longitude uniformly distributed
        lon = jax.random.uniform(k2, (n_samples,), minval=-jnp.pi, maxval=jnp.pi)
        
        return jnp.column_stack([lat, lon])

