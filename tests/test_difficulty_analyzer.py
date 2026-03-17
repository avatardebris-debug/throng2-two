"""Unit tests for mario_difficulty_analyzer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_difficulty_analyzer import (
    DifficultyAnalyzer, HotspotDrillCurriculum, make_practice_level, Hotspot,
)

gen = MarioLevelGenerator(seed=42)
sim = gen.generate(tier=3)
assert sim is not None, "Level generation failed"
print(f"Level: width={sim.width}")

# --- Test 1: Empty death log ---
print("\n=== Test 1: Empty death log ===")
analyzer = DifficultyAnalyzer()
hotspots = analyzer.analyze()
assert hotspots == [], f"Expected [], got {hotspots}"
print("  PASS")

# --- Test 2: Basic clustering ---
print("\n=== Test 2: Basic clustering ===")
deaths = [(15, 12, 2)] * 10 + [(16, 12, 5)] * 5 + [(40, 12, 3)] * 8
analyzer2 = DifficultyAnalyzer(death_log=deaths, cluster_radius=4, min_deaths_for_hotspot=3)
hotspots2 = analyzer2.analyze()
print(f"  Hotspots: {hotspots2}")
assert len(hotspots2) == 2, f"Expected 2 clusters, got {len(hotspots2)}"
assert hotspots2[0].death_count >= hotspots2[1].death_count, "Should be sorted by count"
assert hotspots2[0].death_count == 15, f"Top hotspot should have 15, got {hotspots2[0].death_count}"
print("  PASS")

# --- Test 3: Action analysis ---
print("\n=== Test 3: Action analysis ===")
h = hotspots2[0]  # col 15-16 cluster
assert h.most_common_action == 2, f"Most common action should be 2 (RIGHT), got {h.most_common_action}"
assert 2 in h.action_counts
assert 5 in h.action_counts
print(f"  Action counts: {h.action_counts}")
print("  PASS")

# --- Test 4: Practice level creation ---
print("\n=== Test 4: Practice level creation ===")
practice = make_practice_level(sim, center_col=10, approach_cols=6, challenge_cols=10)
assert practice is not None, "Practice level should be created"
expected_width = min(6 + 10 + 1, sim.width - 1 - max(0, 10-6) + max(0,10-6) + 10 + 1)
print(f"  Practice width: {practice.width}, height: {practice.GRID_H}")
assert practice.width >= 8, f"Practice level too narrow: {practice.width}"
# Mario should be placed near the start
assert practice.mario_col <= 6, f"Mario should start near left, got {practice.mario_col}"
print("  PASS")

# --- Test 5: Practice level near edges ---
print("\n=== Test 5: Practice level near level edges ===")
# Near left edge (col 2)
p_left = make_practice_level(sim, center_col=2, approach_cols=8, challenge_cols=10)
# Right edge
p_right = make_practice_level(sim, center_col=sim.width-3, approach_cols=8, challenge_cols=10)
print(f"  Left-edge practice: {p_left.width if p_left else 'None'}")
print(f"  Right-edge practice: {p_right.width if p_right else 'None'}")
# Both should either succeed or gracefully return None
print("  PASS")

# --- Test 6: HotspotDrillCurriculum lifecycle ---
print("\n=== Test 6: HotspotDrillCurriculum lifecycle ===")
# Use the death-heavy analyzer
drill = HotspotDrillCurriculum(
    source_sim=sim,
    analyzer=analyzer2,
    pass_threshold=0.7,
    min_attempts=5,
)
assert not drill.all_mastered(), "Should not be mastered yet"

practice_sim, info = drill.get_episode()
print(f"  Episode info: hotspot_col={info['hotspot_col']}, deaths={info['death_count']}")
assert practice_sim is not None, "Practice sim should be available"
print(f"  Practice sim: width={practice_sim.width}, mario=({practice_sim.mario_row},{practice_sim.mario_col})")
print("  PASS")

# --- Test 7: Mastery progression ---
print("\n=== Test 7: Mastery progression ===")
# Report enough successes to master the first hotspot
for _ in range(5):
    drill.report_result(success=True)
s = drill.stats()
print(f"  Stats: {s}")
assert s["mastered"] >= 1, f"Should have mastered at least 1 hotspot, got {s}"
print("  PASS")

# --- Test 8: Drill report ---
print("\n=== Test 8: Report ===")
rpt = drill.report()
print(rpt)
assert "MASTERED" in rpt or "deaths" in rpt
print("  PASS")

# --- Test 9: Analyzer report ---
print("\n=== Test 9: Analyzer report ===")
rpt2 = analyzer2.report()
print(rpt2)
assert "hotspot" in rpt2.lower() or "col" in rpt2.lower()
print("  PASS")

print("\n=== ALL 9 TESTS PASSED ===")
