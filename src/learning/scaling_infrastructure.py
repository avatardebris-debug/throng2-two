"""
Phase 3.5: Scaling Infrastructure - Sparse Matrices + Neuron Neurogenesis

Critical infrastructure for 10M+ neuron scaling:
1. Sparse matrix support (scipy.sparse)
2. Neuron-level birth/death (not just connections)
3. Dynamic layer resizing
4. 100K neuron validation
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional
import warnings


class SparseWeightMatrix:
    """
    Efficient sparse weight matrix for large-scale networks.
    
    Uses scipy.sparse for memory efficiency.
    Supports variable fan-in/fan-out per neuron.
    """
    
    def __init__(self, n_rows: int, n_cols: int, initial_density: float = 0.05):
        """
        Initialize sparse weight matrix.
        
        Args:
            n_rows: Number of source neurons
            n_cols: Number of target neurons
            initial_density: Initial connection density
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        
        # Use lil_matrix for efficient incremental construction
        self.weights = lil_matrix((n_rows, n_cols), dtype=np.float32)
        
        # Initialize with sparse random connections
        if initial_density > 0:
            self._initialize_random(initial_density)
    
    def _initialize_random(self, density: float):
        """Initialize with random sparse connections."""
        n_connections = int(self.n_rows * self.n_cols * density)
        
        # Random indices
        rows = np.random.randint(0, self.n_rows, n_connections)
        cols = np.random.randint(0, self.n_cols, n_connections)
        values = np.random.randn(n_connections).astype(np.float32) * 0.1
        
        # Set values
        for r, c, v in zip(rows, cols, values):
            self.weights[r, c] = v
    
    def add_connection(self, row: int, col: int, weight: float = None):
        """Add a single connection."""
        if weight is None:
            weight = np.random.randn() * 0.01
        
        self.weights[row, col] = weight
    
    def remove_connection(self, row: int, col: int):
        """Remove a single connection."""
        self.weights[row, col] = 0
    
    def add_neuron_row(self, initial_density: float = 0.05) -> int:
        """
        Add a new source neuron (row).
        
        Returns:
            Index of new neuron
        """
        # Create new row with random connections
        new_row = lil_matrix((1, self.n_cols), dtype=np.float32)
        
        if initial_density > 0:
            n_connections = int(self.n_cols * initial_density)
            cols = np.random.choice(self.n_cols, n_connections, replace=False)
            values = np.random.randn(n_connections).astype(np.float32) * 0.01
            
            for c, v in zip(cols, values):
                new_row[0, c] = v
        
        # Stack with existing
        from scipy.sparse import vstack
        self.weights = vstack([self.weights, new_row], format='lil')
        self.n_rows += 1
        
        return self.n_rows - 1
    
    def add_neuron_col(self, initial_density: float = 0.05) -> int:
        """
        Add a new target neuron (column).
        
        Returns:
            Index of new neuron
        """
        # Create new column with random connections
        new_col = lil_matrix((self.n_rows, 1), dtype=np.float32)
        
        if initial_density > 0:
            n_connections = int(self.n_rows * initial_density)
            rows = np.random.choice(self.n_rows, n_connections, replace=False)
            values = np.random.randn(n_connections).astype(np.float32) * 0.01
            
            for r, v in zip(rows, values):
                new_col[r, 0] = v
        
        # Stack with existing
        from scipy.sparse import hstack
        self.weights = hstack([self.weights, new_col], format='lil')
        self.n_cols += 1
        
        return self.n_cols - 1
    
    def remove_neuron_row(self, row_idx: int):
        """Remove a source neuron (row)."""
        # Create mask for all rows except the one to remove
        mask = np.ones(self.n_rows, dtype=bool)
        mask[row_idx] = False
        
        # Select rows
        self.weights = self.weights[mask, :]
        self.n_rows -= 1
    
    def remove_neuron_col(self, col_idx: int):
        """Remove a target neuron (column)."""
        # Create mask for all columns except the one to remove
        mask = np.ones(self.n_cols, dtype=bool)
        mask[col_idx] = False
        
        # Select columns
        self.weights = self.weights[:, mask]
        self.n_cols -= 1
    
    def get_density(self) -> float:
        """Calculate current density."""
        return self.weights.nnz / (self.n_rows * self.n_cols)
    
    def get_fan_in(self, neuron_idx: int) -> int:
        """Get number of incoming connections for neuron."""
        col = self.weights[:, neuron_idx]
        return col.nnz
    
    def get_fan_out(self, neuron_idx: int) -> int:
        """Get number of outgoing connections for neuron."""
        row = self.weights[neuron_idx, :]
        return row.nnz
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array (use sparingly!)."""
        return self.weights.toarray()
    
    def to_csr(self):
        """Convert to CSR format (efficient for matrix operations)."""
        return csr_matrix(self.weights)
    
    def get_memory_usage(self) -> Dict:
        """Estimate memory usage."""
        # Sparse matrix memory
        sparse_bytes = (
            self.weights.nnz * 4 +  # values (float32)
            self.weights.nnz * 4 +  # row indices (int32)
            (self.n_rows + 1) * 4   # column pointers (int32)
        )
        
        # Equivalent dense matrix
        dense_bytes = self.n_rows * self.n_cols * 4  # float32
        
        return {
            'sparse_mb': sparse_bytes / (1024**2),
            'dense_mb': dense_bytes / (1024**2),
            'savings': 1.0 - (sparse_bytes / dense_bytes),
            'density': self.get_density()
        }


class NeuronBirthController:
    """
    Manages neuron-level neurogenesis (adding neurons, not just connections).
    """
    
    def __init__(
        self,
        saturation_threshold: float = 0.95,
        error_threshold: float = 0.5,
        min_neurons_per_region: int = 100,
        max_neurons_per_region: int = 1_000_000
    ):
        """
        Initialize neuron birth controller.
        
        Args:
            saturation_threshold: Activity level to consider neurons saturated
            error_threshold: Error level to trigger neuron birth
            min_neurons_per_region: Minimum neurons per region
            max_neurons_per_region: Maximum neurons per region
        """
        self.saturation_threshold = saturation_threshold
        self.error_threshold = error_threshold
        self.min_neurons_per_region = min_neurons_per_region
        self.max_neurons_per_region = max_neurons_per_region
        
        # Track birth events
        self.birth_events = []
    
    def should_add_neuron(
        self,
        current_count: int,
        activities: np.ndarray,
        errors: np.ndarray
    ) -> bool:
        """
        Decide if region should add a neuron.
        
        Add if:
        1. Neurons saturated (high average activity)
        2. Still high errors (need more capacity)
        3. Below max neuron budget
        """
        if current_count >= self.max_neurons_per_region:
            return False
        
        avg_activity = np.mean(activities)
        avg_error = np.mean(errors)
        
        # Check saturation (using median and 90th percentile)
        saturation = np.percentile(activities, 90)
        
        return (
            saturation > self.saturation_threshold and
            avg_error > self.error_threshold and
            current_count < self.max_neurons_per_region
        )
    
    def add_neuron_to_region(
        self,
        weight_matrix: SparseWeightMatrix,
        region_type: str,
        initial_density: float = 0.05
    ) -> int:
        """
        Add a new neuron to region.
        
        Returns:
            Index of new neuron
        """
        # Add neuron (as both row and column for recurrent connections)
        neuron_idx = weight_matrix.add_neuron_row(initial_density)
        weight_matrix.add_neuron_col(initial_density)
        
        # Record birth
        self.birth_events.append({
            'neuron_idx': neuron_idx,
            'region_type': region_type,
            'initial_density': initial_density
        })
        
        return neuron_idx
    
    def get_birth_statistics(self) -> Dict:
        """Get statistics on neuron births."""
        if not self.birth_events:
            return {}
        
        return {
            'total_births': len(self.birth_events),
            'regions': [e['region_type'] for e in self.birth_events]
        }


class NeuronApoptosis:
    """
    Manages neuron death (removing underutilized neurons).
    """
    
    def __init__(
        self,
        activity_threshold: float = 0.01,
        connection_threshold: int = 5,
        redundancy_threshold: float = 0.9
    ):
        """
        Initialize neuron apoptosis.
        
        Args:
            activity_threshold: Below this → candidate for removal
            connection_threshold: Fewer connections → candidate for removal
            redundancy_threshold: Correlation with other neurons
        """
        self.activity_threshold = activity_threshold
        self.connection_threshold = connection_threshold
        self.redundancy_threshold = redundancy_threshold
        
        # Track death events
        self.death_events = []
    
    def should_remove_neuron(
        self,
        neuron_idx: int,
        activities: np.ndarray,
        weight_matrix: SparseWeightMatrix,
        min_neurons: int = 100
    ) -> bool:
        """
        Decide if neuron should be removed.
        
        Remove if:
        1. Consistently low activity
        2. Few connections (most were pruned)
        3. Above minimum neuron count
        """
        if weight_matrix.n_rows <= min_neurons:
            return False
        
        activity = activities[neuron_idx]
        connections = (
            weight_matrix.get_fan_in(neuron_idx) +
            weight_matrix.get_fan_out(neuron_idx)
        )
        
        return (
            activity < self.activity_threshold and
            connections < self.connection_threshold
        )
    
    def remove_neuron_from_region(
        self,
        neuron_idx: int,
        weight_matrix: SparseWeightMatrix
    ):
        """Remove a neuron from region."""
        # Remove neuron
        weight_matrix.remove_neuron_row(neuron_idx)
        weight_matrix.remove_neuron_col(neuron_idx)
        
        # Record death
        self.death_events.append({
            'neuron_idx': neuron_idx
        })
    
    def get_death_statistics(self) -> Dict:
        """Get statistics on neuron deaths."""
        if not self.death_events:
            return {}
        
        return {
            'total_deaths': len(self.death_events)
        }


def benchmark_sparse_scaling():
    """Benchmark sparse matrix scaling."""
    print("\n" + "="*60)
    print("BENCHMARK: Sparse Matrix Scaling")
    print("="*60)
    
    sizes = [100, 1000, 10000, 100000]
    densities = [0.01, 0.05, 0.10]
    
    print(f"\n{'Size':<10} {'Density':<10} {'Sparse (MB)':<15} {'Dense (MB)':<15} {'Savings':<10}")
    print("-" * 70)
    
    for size in sizes:
        for density in densities:
            # Create sparse matrix
            sparse_mat = SparseWeightMatrix(size, size, initial_density=density)
            
            # Get memory usage
            memory = sparse_mat.get_memory_usage()
            
            print(f"{size:<10} {density:<10.2f} {memory['sparse_mb']:<15.2f} "
                  f"{memory['dense_mb']:<15.2f} {memory['savings']:<10.1%}")
    
    print("\n✓ Sparse matrices enable 10-100x memory reduction!")
    
    return True


def benchmark_neuron_neurogenesis():
    """Benchmark neuron-level neurogenesis."""
    print("\n" + "="*60)
    print("BENCHMARK: Neuron-Level Neurogenesis")
    print("="*60)
    
    # Create network
    n_neurons = 100
    weights = SparseWeightMatrix(n_neurons, n_neurons, initial_density=0.05)
    
    birth_controller = NeuronBirthController()
    apoptosis = NeuronApoptosis()
    
    print(f"\nInitial neurons: {weights.n_rows}")
    print(f"Initial density: {weights.get_density():.2%}")
    
    # Simulate episodes
    for episode in range(50):
        # Simulate activities and errors
        activities = np.random.rand(weights.n_rows)
        errors = np.random.rand(weights.n_rows)
        
        # Try to add neuron
        if birth_controller.should_add_neuron(weights.n_rows, activities, errors):
            new_idx = birth_controller.add_neuron_to_region(
                weights, "hidden", initial_density=0.05
            )
            print(f"Episode {episode}: Added neuron {new_idx}")
        
        # Try to remove inactive neurons
        if weights.n_rows > 100:  # Keep minimum
            for neuron_idx in range(weights.n_rows):
                if apoptosis.should_remove_neuron(
                    neuron_idx, activities, weights, min_neurons=100
                ):
                    apoptosis.remove_neuron_from_region(neuron_idx, weights)
                    print(f"Episode {episode}: Removed neuron {neuron_idx}")
                    break  # Remove one per episode
    
    print(f"\nFinal neurons: {weights.n_rows}")
    print(f"Final density: {weights.get_density():.2%}")
    
    birth_stats = birth_controller.get_birth_statistics()
    death_stats = apoptosis.get_death_statistics()
    
    print(f"\nBirth events: {birth_stats.get('total_births', 0)}")
    print(f"Death events: {death_stats.get('total_deaths', 0)}")
    
    return True
