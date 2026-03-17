"""Quick unit test for ZoneCurriculum logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_zone_curriculum import ZoneCurriculum, make_zones

gen = MarioLevelGenerator(seed=42)
cur = ZoneCurriculum(generator=gen, tier=3, seed=42)

# Test 1: Zone creation
print("=== Test 1: Zone creation ===")
for z in cur.zones:
    print("  ", z)
w = cur._current_level.width
print("  Level width:", w)
assert len(cur.zones) == 6, "Expected 6 zones"
assert cur.zones[-1].target_col == w - 1, "Last zone should target end of level"
print("  PASS")

# Test 2: Episode lifecycle
print("\n=== Test 2: Episode lifecycle ===")
sim, info = cur.get_episode()
print("  Zone:", info["zone_name"], "target:", info["target_col"])
print("  Mario start:", sim.mario_row, sim.mario_col)
assert info["zone_id"] == 0, "Should start at zone 0"
print("  PASS")

# Test 3: Column novelty
print("\n=== Test 3: Column novelty ===")
b1 = cur.column_visited(5)
b2 = cur.column_visited(5)
b3 = cur.column_visited(6)
print("  First visit col 5:", round(b1, 3), "(expect", cur.novelty_bonus, ")")
print("  Same col same ep:", round(b2, 3), "(expect 0)")
print("  New col 6:", round(b3, 3), "(expect", cur.novelty_bonus, ")")
assert b1 > 0, "First visit should give bonus"
assert b2 == 0, "Same col same episode should give 0"
assert b3 > 0, "New col should give bonus"
print("  PASS")

# Test 4: Promotion
print("\n=== Test 4: Promotion ===")
initial_zone = cur.current_zone_idx
# Use the level width as final_col — this always succeeds regardless of zone
level_end = cur._current_level.width
for _ in range(100):
    cur.report_result(final_col=level_end, won=False, alive=True)
s = cur.stats()
print("  Zone after 100 successes:", s["current_zone"])
print("  Promotions:", s["promotions"])
assert s["current_zone"] > initial_zone, "Should have promoted"
print("  PASS")

# Test 5: Demotion
print("\n=== Test 5: Demotion ===")
promoted_zone = cur.current_zone_idx
demotions_before = len(cur._demotions)
print("  Promoted to zone:", promoted_zone, "demotions before:", demotions_before)
for _ in range(60):
    cur.report_result(final_col=0, won=False, alive=False)
s2 = cur.stats()
demotions_after = s2["demotions"]
print("  Zone after 60 failures:", s2["current_zone"])
print("  Demotions:", demotions_after)
assert demotions_after > demotions_before, "At least one demotion should have occurred"
print("  PASS")

# Test 6: Death hotspots
print("\n=== Test 6: Death hotspots ===")
for _ in range(20):
    cur.record_death(col=15, row=12, last_action=2)
for _ in range(5):
    cur.record_death(col=40, row=12, last_action=5)
hotspots = cur.get_death_hotspots()
print("  Hotspots:", hotspots)
assert len(hotspots) >= 2, "Should have at least 2 hotspot clusters"
assert hotspots[0]["count"] >= 20, "Top hotspot should have >= 20 deaths"
print("  PASS")

# Test 7: Zone summary
print("\n=== Test 7: Zone summary ===")
summary = cur.zone_summary()
print(summary)
assert "Zone 0" in summary
assert "Zone 5" in summary
print("  PASS")

print("\n=== ALL 7 TESTS PASSED ===")
