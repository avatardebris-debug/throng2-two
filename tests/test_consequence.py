"""Test ConsequenceCascadeBuffer + UncertaintySeeker + PrecursorGraph."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import numpy as np

    Z = 16
    N_ACT = 4

    print("=== Test 1: ConsequenceCascadeBuffer ===")
    from src.learning.consequence_cascade import ConsequenceCascadeBuffer

    cascade = ConsequenceCascadeBuffer(capacity=5000, decay=0.995, min_magnitude=0.5)

    # Simulate an episode with a rare consequence at step 99
    cascade.begin_episode()
    for i in range(100):
        s = np.random.randn(Z).astype('float32') * 0.1
        ns = s + np.random.randn(Z).astype('float32') * 0.01
        reward = 0.0 if i < 99 else 10.0  # Huge reward on last step
        cascade.store_step(s, i % N_ACT, ns, reward, entity_tag='player')

    # Mark the reward step as consequential
    cascade.mark_consequential(step=99, magnitude=10.0, tag="win")
    cascade.end_episode()

    print(f"  Buffer size: {len(cascade)}")
    print(f"  Stats: {cascade.stats()}")

    # Sample — should heavily favor steps near 99
    batch = cascade.sample(32)
    avg_weight = np.mean([t.weight for t in batch])
    max_weight = max(t.weight for t in batch)
    print(f"  Sample: avg_weight={avg_weight:.4f}, max_weight={max_weight:.4f}")
    print(f"  High-weight transitions dominate: {sum(1 for t in batch if t.weight > 0.1)}/32")
    print()

    # Test auto-detect consequences
    cascade2 = ConsequenceCascadeBuffer(capacity=5000, min_magnitude=2.0)
    cascade2.begin_episode()
    for i in range(50):
        r = 5.0 if i == 30 else 0.0  # spike at step 30
        cascade2.store_step(np.zeros(Z), i % N_ACT, np.zeros(Z), r)
    cascade2.end_episode(auto_detect_consequences=True)
    print(f"  Auto-detect: {cascade2.stats()['n_consequences']} consequences found (expect 1)")
    print()

    print("=== Test 2: PrecursorGraph ===")
    from src.learning.uncertainty_seeker import PrecursorGraph

    graph = PrecursorGraph(min_edge_strength=0.05)

    # Simulate: interacting with 'key' improves 'door' confidence
    for _ in range(10):
        graph.observe_confidence_change('key', 'door', 0.15 + np.random.randn() * 0.02)
        graph.observe_confidence_change('switch', 'wall', 0.10 + np.random.randn() * 0.02)
        graph.observe_confidence_change('key', 'wall', -0.02 + np.random.randn() * 0.01)  # noise

    graph.rebuild()
    print(f"  Edges: {graph.all_edges()}")
    print(f"  Precursors of 'door': {graph.get_precursors('door')}")
    print(f"  Strongest precursor of 'door': {graph.strongest_precursor('door')}")
    assert graph.strongest_precursor('door') == 'key'
    print(f"  Precursors of 'wall': {graph.get_precursors('wall')}")
    assert graph.strongest_precursor('wall') == 'switch'
    print(f"  Stats: {graph.stats()}")
    print()

    print("=== Test 3: UncertaintySeeker ===")
    from src.learning.uncertainty_seeker import UncertaintySeeker
    from src.cell.world_model import CellWorldModel

    wm = CellWorldModel(feature_dim=Z, n_actions=N_ACT, hidden_size=32,
                        min_transitions=5, batch_size=4)

    # Seed WM
    for i in range(30):
        s = np.random.randn(Z).astype('float32') * 0.1
        ns = s + np.random.randn(Z).astype('float32') * 0.02
        wm.store_transition(s, i % N_ACT, ns, 0.1)
        wm.train_step()

    seeker = UncertaintySeeker(wm, n_actions=N_ACT, understood_threshold=0.10,
                               max_investigation=5)

    # Register entities
    for tag in ['player', 'goomba', 'door', 'key', 'switch']:
        seeker.register_entity(tag, is_consequential=(tag in ['door']))

    # 1. First target should be UNKNOWN entity
    target = seeker.next_target()
    print(f"  First target: {target} (should be an UNKNOWN entity)")
    assert target is not None

    # 2. Interact with entities → transition through states
    s = np.random.randn(Z).astype('float32') * 0.1
    for i in range(20):
        entity = ['player', 'goomba', 'key', 'switch', 'door'][i % 5]
        ns = s + np.random.randn(Z).astype('float32') * (0.02 if entity != 'door' else 0.5)
        seeker.observe(s, i % N_ACT, ns, entity_tag=entity, reward=0.0)
        s = ns

    print("  After 20 interactions:")
    for tag, rec in seeker._entities.items():
        print(f"    {tag:10s}: status={rec.status.value:15s} "
              f"err={rec.recent_avg_error:.3f} interactions={rec.n_interactions}")

    # 3. Check exploration priorities
    priorities = seeker.get_exploration_priorities()
    print(f"  Priorities: { {k: round(v, 2) for k, v in sorted(priorities.items(), key=lambda x: -x[1])} }")

    # 4. Simulate precursor relationship: after 'key', 'door' gets more predictable
    for _ in range(5):
        seeker.precursor_graph.observe_confidence_change('key', 'door', 0.12)
    seeker.precursor_graph.rebuild()
    chain = seeker.get_precursor_chain('door')
    print(f"  Precursor chain for 'door': {chain}")

    # 5. Stochastic entities
    print(f"  Stochastic: {seeker.get_stochastic_entities()}")

    print(f"  Full stats: {seeker.stats()}")
    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
