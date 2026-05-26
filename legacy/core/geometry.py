"""
Sacred Geometry - Optimal neuron placement and connection patterns.

Uses principles from nature:
- Fibonacci spiral (golden ratio)
- Small-world topology (local + sparse long-range)
- Wiring cost minimization
"""

import numpy as np
from typing import Tuple, List
from scipy.sparse import csr_matrix, random as sparse_random


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """
    Distribute points evenly on a sphere using golden ratio.
    
    This is optimal 3D point distribution - appears in sunflower seeds,
    pinecones, nautilus shells, etc.
    
    Args:
        n_points: Number of points to place
        
    Returns:
        Array of shape (n_points, 3) with x, y, z coordinates
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    
    points = []
    for i in range(n_points):
        # Vertical position
        y = 1 - (i / (n_points - 1)) * 2  # y from 1 to -1
        
        # Radius at this height
        radius = np.sqrt(1 - y**2)
        
        # Angle using golden ratio
        theta = 2 * np.pi * i / phi
        
        # Convert to Cartesian
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
        
    return np.array(points)


def fibonacci_spiral_2d(n_points: int) -> np.ndarray:
    """
    2D spiral using golden ratio (like sunflower seed pattern).
    
    Args:
        n_points: Number of points to place
        
    Returns:
        Array of shape (n_points, 2) with x, y coordinates
    """
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi)  # ≈ 137.5 degrees
    
    points = []
    for i in range(n_points):
        # Distance from center (sqrt for even spacing)
        r = np.sqrt(i / n_points)
        
        # Angle
        theta = i * golden_angle
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        points.append([x, y])
        
    return np.array(points)


def place_neurons(n_neurons: int, dimension: int = 2) -> np.ndarray:
    """
    Place neurons optimally in space.
    
    Args:
        n_neurons: Number of neurons
        dimension: 2 or 3
        
    Returns:
        Neuron positions
    """
    if dimension == 2:
        return fibonacci_spiral_2d(n_neurons)
    elif dimension == 3:
        return fibonacci_sphere(n_neurons)
    else:
        raise ValueError("Dimension must be 2 or 3")


def small_world_connections(positions: np.ndarray,
                           connection_prob: float = 0.02,
                           local_radius: float = 0.3,
                           long_range_prob: float = 0.005) -> np.ndarray:
    """
    Create small-world network topology.
    
    Combines:
    - High local clustering (nearby neurons connect)
    - Short path length (sparse long-range shortcuts)
    
    This is the brain's topology: efficient wiring + fast global communication.
    
    Args:
        positions: Neuron positions (n_neurons, dim)
        connection_prob: Overall connection probability
        local_radius: Distance for local connections
        long_range_prob: Probability of long-range shortcuts
        
    Returns:
        Connection matrix (n_neurons, n_neurons)
    """
    n_neurons = len(positions)
    connections = np.zeros((n_neurons, n_neurons))
    
    # Compute all pairwise distances
    distances = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Create connections
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            dist = distances[i, j]
            
            # Local connections (high probability if close)
            if dist < local_radius:
                if np.random.random() < connection_prob * 5:  # 5x more likely
                    weight = np.random.uniform(0.1, 0.5) * (1 - dist/local_radius)
                    connections[i, j] = weight
                    connections[j, i] = weight
                    
            # Long-range shortcuts (power-law probability)
            elif np.random.random() < long_range_prob * (dist ** -2):
                weight = np.random.uniform(0.05, 0.2)
                connections[i, j] = weight
                connections[j, i] = weight
                
    return connections


def small_world_connections_sparse(positions: np.ndarray,
                                   connection_prob: float = 0.02,
                                   local_radius: float = 0.3,
                                   long_range_prob: float = 0.005):
    """
    Create small-world network topology using EFFICIENT sparse matrices.
    
    This version avoids nested Python loops and is suitable for large networks
    (100K+ neurons). Uses vectorized operations for speed.
    
    Args:
        positions: Neuron positions (n_neurons, dim)
        connection_prob: Overall connection probability
        local_radius: Distance for local connections
        long_range_prob: Probability of long-range shortcuts
        
    Returns:
        Sparse CSR connection matrix (n_neurons, n_neurons)
    """
    n_neurons = len(positions)
    
    # For large networks (>10K), use pure random sparse structure
    # This is much faster than computing distances
    if n_neurons > 10000:
        print(f"  Using fast random sparse structure for {n_neurons:,} neurons...")
        density = min(connection_prob, 0.001)
        
        # Build sparse matrix in chunks to avoid memory issues
        from scipy.sparse import coo_matrix, vstack
        n_connections_total = int(n_neurons * n_neurons * density)
        
        # Limit connections to avoid memory issues
        max_connections = min(n_connections_total, 100_000_000)  # Cap at 100M connections
        actual_density = max_connections / (n_neurons * n_neurons)
        
        print(f"  Target connections: {max_connections:,} (density: {actual_density:.6f})")
        
        # Build in chunks
        chunk_size = 10_000_000  # 10M connections per chunk
        chunks = []
        
        for i in range(0, max_connections, chunk_size):
            end = min(i + chunk_size, max_connections)
            n_conn = end - i
            
            # Random indices for this chunk
            rows = np.random.randint(0, n_neurons, n_conn, dtype=np.int32)
            cols = np.random.randint(0, n_neurons, n_conn, dtype=np.int32)
            data = np.random.uniform(0, 0.3, n_conn).astype(np.float32)
            
            chunk = coo_matrix((data, (rows, cols)), shape=(n_neurons, n_neurons), dtype=np.float32)
            chunks.append(chunk)
        
        # Combine chunks
        if len(chunks) == 1:
            connections = chunks[0]
        else:
            # Sum all chunks
            connections = chunks[0]
            for chunk in chunks[1:]:
                connections = connections + chunk
        
        # Make symmetric
        connections = (connections + connections.T) / 2
        
        # Convert to CSR for efficient operations
        return connections.tocsr()
    
    # For medium networks, use chunked approach
    from scipy.sparse import lil_matrix
    connections = lil_matrix((n_neurons, n_neurons), dtype=np.float32)
    
    # Add local connections in chunks to avoid memory issues
    chunk_size = min(1000, n_neurons)
    
    for i in range(0, n_neurons, chunk_size):
        end_i = min(i + chunk_size, n_neurons)
        
        for local_i in range(i, end_i):
            # Compute distances from this neuron to a subset of others
            for j in range(local_i + 1, min(local_i + 5000, n_neurons)):
                dist = np.linalg.norm(positions[local_i] - positions[j])
                
                # Local connection
                if dist < local_radius and np.random.random() < connection_prob * 5:
                    weight = np.random.uniform(0.1, 0.5) * (1 - dist/local_radius)
                    connections[local_i, j] = weight
                    connections[j, local_i] = weight
                # Long-range connection
                elif np.random.random() < long_range_prob * 0.1:
                    weight = np.random.uniform(0.05, 0.2)
                    connections[local_i, j] = weight
                    connections[j, local_i] = weight
    
    # Convert to CSR for efficient operations
    return connections.tocsr()



def wiring_cost(positions: np.ndarray, connections: np.ndarray) -> float:
    """
    Calculate total wiring cost (sum of connection lengths).
    
    Biological brains minimize this - it's metabolically expensive to
    maintain long axons.
    
    Args:
        positions: Neuron positions
        connections: Connection matrix
        
    Returns:
        Total wiring length
    """
    total_cost = 0.0
    n_neurons = len(positions)
    
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if connections[i, j] > 0:
                dist = np.linalg.norm(positions[i] - positions[j])
                total_cost += dist * connections[i, j]
                
    return total_cost


def connection_statistics(connections: np.ndarray) -> dict:
    """
    Compute network statistics.
    
    Args:
        connections: Connection matrix
        
    Returns:
        Dictionary of statistics
    """
    n_neurons = len(connections)
    n_connections = np.sum(connections > 0) / 2  # Divide by 2 (symmetric)
    
    # Degree distribution
    degrees = np.sum(connections > 0, axis=1)
    
    # Clustering coefficient (local clustering)
    clustering = 0.0
    for i in range(n_neurons):
        neighbors = np.where(connections[i] > 0)[0]
        if len(neighbors) < 2:
            continue
        # How many neighbor pairs are connected?
        neighbor_connections = 0
        for j in neighbors:
            for k in neighbors:
                if j < k and connections[j, k] > 0:
                    neighbor_connections += 1
        max_connections = len(neighbors) * (len(neighbors) - 1) / 2
        if max_connections > 0:
            clustering += neighbor_connections / max_connections
    clustering /= n_neurons
    
    return {
        'n_neurons': n_neurons,
        'n_connections': int(n_connections),
        'connection_density': n_connections / (n_neurons * (n_neurons - 1) / 2),
        'avg_degree': np.mean(degrees),
        'clustering_coefficient': clustering,
        'avg_weight': np.mean(connections[connections > 0])
    }
