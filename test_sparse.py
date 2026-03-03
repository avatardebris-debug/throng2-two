"""Simple test to isolate the sparse matrix issue."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.sparse import random as sparse_random

print("Testing sparse matrix creation...")

# Test 1: Simple sparse matrix
print("\n1. Creating small sparse matrix...")
m1 = sparse_random(1000, 1000, density=0.01, format='csr')
print(f"   Created: {m1.shape}, dtype={m1.dtype}, nnz={m1.nnz}")

# Test 2: Large sparse matrix
print("\n2. Creating 1M x 1M sparse matrix...")
m2 = sparse_random(1_000_000, 1_000_000, density=0.0001, format='csr')
print(f"   Created: {m2.shape}, dtype={m2.dtype}, nnz={m2.nnz}")

# Test 3: Convert to float32
print("\n3. Converting to float32...")
m3 = m2.astype(np.float32)
print(f"   Converted: {m3.shape}, dtype={m3.dtype}")

# Test 4: Modify data
print("\n4. Modifying data...")
m3.data[:] = np.abs(m3.data) * 0.3
print(f"   Modified: min={m3.data.min():.4f}, max={m3.data.max():.4f}")

# Test 5: Make symmetric
print("\n5. Making symmetric...")
m4 = (m3 + m3.T) / 2
print(f"   Symmetric: {m4.shape}, dtype={m4.dtype}, nnz={m4.nnz}")

print("\n✓ All sparse matrix operations successful!")
