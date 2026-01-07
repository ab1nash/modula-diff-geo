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

