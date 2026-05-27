"""
test_go_explore_adapter.py — Tests for ZCellArchive, GoExploreRunner, GoExploreMetaRouter
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.encoder.go_explore_adapter import ZCellArchive, GoExploreRunner, GoExploreMetaRouter, CellEntry


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: ZCellArchive — add, accept, sample
# ──────────────────────────────────────────────────────────────────────────────

def test_archive_add_and_accept():
    print("\n=== Test 1: ZCellArchive add/accept logic ===")
    arch = ZCellArchive(z_dim=4, resolution=4)
    rng = np.random.RandomState(0)

    z1 = rng.randn(4).astype(np.float32)
    z1 /= np.linalg.norm(z1) + 1e-8

    # Add initial cell
    added = arch.add(z1, score=1.0, trajectory_len=10)
    assert added, "First add should succeed"
    assert arch.size == 1

    # Same cell with better score → should update
    added2 = arch.add(z1, score=2.0, trajectory_len=10)
    assert added2, "Better score should update"
    key = arch.cell_key(z1)
    assert arch._archive[key].score == 2.0

    # Same cell with worse score → should reject
    added3 = arch.add(z1, score=0.5, trajectory_len=5)
    assert not added3, "Worse score should be rejected"

    # Same score, shorter trajectory → should update
    added4 = arch.add(z1, score=2.0, trajectory_len=3)
    assert added4, "Same score, shorter path should update"
    assert arch._archive[key].trajectory_len == 3

    print(f"  Archive stats: {arch.stats()}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Novelty-weighted sampling — rarely-visited cells win
# ──────────────────────────────────────────────────────────────────────────────

def test_novelty_weighted_sampling():
    print("\n=== Test 2: Novelty-weighted sampling ===")
    arch = ZCellArchive(z_dim=4, resolution=1)  # low resolution = large cells
    rng = np.random.RandomState(42)

    # Two distinct cells
    z_common = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    z_novel  = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    arch.add(z_common, score=1.0, trajectory_len=5)
    arch.add(z_novel,  score=0.5, trajectory_len=5)

    # Pre-visit z_common many times to make it less novel
    k_common = arch.cell_key(z_common)
    arch._archive[k_common].nb_visits = 100

    # Sample many times: z_novel should be selected much more often
    counts = {arch.cell_key(z_common): 0, arch.cell_key(z_novel): 0}
    for _ in range(200):
        key = arch.sample_cell(rng)
        if key in counts:
            counts[key] += 1

    k_novel = arch.cell_key(z_novel)
    print(f"  common visits: {counts[arch.cell_key(z_common)]}, novel visits: {counts[k_novel]}")
    assert counts[k_novel] > counts[arch.cell_key(z_common)], \
        "Novel cell should be sampled more often than frequently-visited cell"
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: GoExploreMetaRouter — seeds from most similar game
# ──────────────────────────────────────────────────────────────────────────────

def test_meta_router_picks_closest_game():
    print("\n=== Test 3: MetaRouter picks most similar source game ===")

    # Mock MetaEncoder with a fixed similarity matrix
    class MockMetaEncoder:
        def similarity_matrix(self):
            # "lunar" is most similar to "cartpole", closer than "mario"
            return {
                "lunar": {"cartpole": 0.85, "mario": 0.3},
                "cartpole": {"lunar": 0.85, "mario": 0.25},
                "mario": {"cartpole": 0.25, "lunar": 0.3},
            }

    # Populate archives: cartpole has cells, mario has cells, lunar is empty
    arch_cartpole = ZCellArchive(z_dim=4)
    arch_mario    = ZCellArchive(z_dim=4)
    arch_lunar    = ZCellArchive(z_dim=4)

    rng = np.random.RandomState(1)
    for i in range(10):
        z = rng.randn(4).astype(np.float32)
        arch_cartpole.add(z, score=float(i), trajectory_len=i+1)
    for i in range(5):
        z = rng.randn(4).astype(np.float32)
        arch_mario.add(z, score=float(i), trajectory_len=i+1)

    archives = {"cartpole": arch_cartpole, "mario": arch_mario, "lunar": arch_lunar}
    router = GoExploreMetaRouter(MockMetaEncoder(), archives)

    source = router.best_source_game("lunar", min_cells=5)
    assert source == "cartpole", f"Expected 'cartpole' as closest with cells, got {source!r}"

    # Seed lunar from cartpole
    n_seeded = router.seed_for("lunar", top_k=5)
    assert n_seeded > 0, "Should have seeded some cells into lunar archive"
    assert arch_lunar.size > 0, "Lunar archive should now have cells"

    print(f"  best_source_game='cartpole', seeded {n_seeded} cells into lunar")
    print(f"  Router stats: {router.stats()}")
    print("  PASS")


if __name__ == "__main__":
    test_archive_add_and_accept()
    test_novelty_weighted_sampling()
    test_meta_router_picks_closest_game()
    print("\n=== ALL GO-EXPLORE ADAPTER TESTS PASSED ===")
