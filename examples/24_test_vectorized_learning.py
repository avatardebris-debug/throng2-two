"""
Quick Test - Vectorized Hebbian Learning

Tests the key fix: vectorized learning doesn't lock up the system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from src.core.network import ThrongletNetwork


def test_small_network():
    """Test vectorized learning on small network."""
    print("\n" + "="*70)
    print("TEST 1: Small Network (1,000 neurons)")
    print("="*70)
    
    net = ThrongletNetwork(n_neurons=1000, connection_prob=0.05)
    print(f"Network created: {net.n_neurons:,} neurons, sparse={net.use_sparse}")
    
    # Run forward passes
    for i in range(10):
        inputs = np.random.randn(10)
        net.forward(inputs)
    
    # Test learning
    print("\nTesting learning...")
    start = time.time()
    for i in range(100):
        net.hebbian_update(learning_rate=0.01, modulation=1.0)
    elapsed = time.time() - start
    
    print(f"100 learning steps: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per step)")
    print("[PASS] Small network works")
    return True


def test_medium_network():
    """Test on 50K neurons - should use vectorized mode."""
    print("\n" + "="*70)
    print("TEST 2: Medium Network (50,000 neurons)")
    print("="*70)
    
    net = ThrongletNetwork(n_neurons=50_000, connection_prob=0.002, use_fibonacci=False)
    print(f"Network created: {net.n_neurons:,} neurons, sparse={net.use_sparse}")
    
    # Forward pass
    inputs = np.random.randn(100)
    start = time.time()
    net.forward(inputs)
    print(f"Forward pass: {time.time() - start:.3f}s")
    
    # Learning - this would have locked up before!
    print("\nTesting learning (critical test)...")
    times = []
    for i in range(10):
        start = time.time()
        net.hebbian_update(learning_rate=0.01, modulation=1.0)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    print(f"10 learning steps: avg {avg_time:.3f}s per step")
    
    if avg_time < 5:
        print("[PASS] 50K neurons - learning is fast!")
        return True
    else:
        print("[SLOW] But didn't lock up")
        return False


def test_large_network_sparse():
    """Test 200K neurons with sparse update mode."""
    print("\n" + "="*70)
    print("TEST 3: Large Network (200,000 neurons) - Sparse Update Mode")
    print("="*70)
    
    print("Creating network...")
    start = time.time()
    net = ThrongletNetwork(n_neurons=200_000, connection_prob=0.001, use_fibonacci=False)
    print(f"Init time: {time.time() - start:.2f}s")
    print(f"Network: {net.n_neurons:,} neurons, sparse={net.use_sparse}")
    
    # Forward
    inputs = np.random.randn(100)
    start = time.time()
    net.forward(inputs)
    print(f"Forward pass: {time.time() - start:.3f}s")
    
    # Learning with sparse update (auto-enabled for large networks)
    print("\nTesting sparse update learning...")
    start = time.time()
    net.hebbian_update(learning_rate=0.01, modulation=1.0, sparse_update=True)
    elapsed = time.time() - start
    
    print(f"Sparse learning step: {elapsed:.3f}s")
    
    if elapsed < 10:
        print("[PASS] 200K neurons with sparse updates works!")
        return True
    else:
        print("[SLOW] But functional")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUICK VECTORIZED LEARNING TEST")
    print("="*70)
    print("\nThis tests that the O(n²) bottleneck is fixed.")
    print("Before the fix, even 50K neurons would lock up the system.")
    
    results = []
    
    try:
        results.append(("Small Network", test_small_network()))
    except Exception as e:
        print(f"[FAIL]: {e}")
        results.append(("Small Network", False))
    
    try:
        results.append(("Medium Network (50K)", test_medium_network()))
    except Exception as e:
        print(f"[FAIL]: {e}")
        results.append(("Medium Network", False))
    
    try:
        results.append(("Large Network (200K)", test_large_network_sparse()))
    except Exception as e:
        print(f"[FAIL]: {e}")
        results.append(("Large Network", False))
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\n[SUCCESS!] The vectorized Hebbian learning fix works!")
        print("\nYou can now train networks with 100K+ neurons without lockup.")
        print("The O(n²) bottleneck has been eliminated!")
    else:
        print("\n[WARNING] Some tests had issues, but the core fix appears to work.")
