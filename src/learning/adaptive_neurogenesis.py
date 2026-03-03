"""
Adaptive Neurogenesis + Compression Lifecycle (Phase 3b)

Key concepts:
1. Dynamic balance: Grow where needed, compress when stable
2. Density targets per region (sensory: 10-20%, hidden: 5-10%, output: 15-25%)
3. Brain lifecycle: Active → Stabilize → Compress → Store → Reactivate
4. Performance-driven: Grow when errors high + novelty high + density low
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pickle


class RegionType(Enum):
    """Brain region types with different density requirements."""
    SENSORY = "sensory"      # Input processing
    HIDDEN = "hidden"        # Internal processing
    OUTPUT = "output"        # Action/decision
    MEMORY = "memory"        # Long-term storage


class LifecyclePhase(Enum):
    """Brain region lifecycle states."""
    ACTIVE = "active"              # Actively learning, uncompressed
    STABILIZING = "stabilizing"    # Activity decreasing, preparing to compress
    COMPRESSED = "compressed"      # Compressed, low activity
    STORED = "stored"              # Saved to disk, not in RAM
    DECOMPRESSING = "decompressing"  # Being loaded back


class AdaptiveDensityController:
    """
    Manages optimal density per brain region.
    
    Density targets:
    - Sensory: 10-20% (rich representations)
    - Hidden: 5-10% (efficient processing)
    - Output: 15-25% (precise control)
    - Memory: 1-5% (very sparse, compressed)
    """
    
    def __init__(self):
        # Density targets (min, max) per region type
        self.density_targets = {
            RegionType.SENSORY: (0.10, 0.20),
            RegionType.HIDDEN: (0.05, 0.10),
            RegionType.OUTPUT: (0.15, 0.25),
            RegionType.MEMORY: (0.01, 0.05)
        }
        
        # Performance thresholds
        self.growth_error_threshold = 0.5      # Grow if error > this
        self.growth_novelty_threshold = 0.7    # Grow if novelty > this
        self.pruning_redundancy_threshold = 0.85
        
    def get_density_range(self, region_type: RegionType) -> Tuple[float, float]:
        """Get density range for region type."""
        return self.density_targets[region_type]
    
    def compute_current_density(self, weights: np.ndarray) -> float:
        """Calculate current connection density."""
        return np.count_nonzero(weights) / weights.size
    
    def should_grow(self,
                   region_type: RegionType,
                   current_density: float,
                   prediction_error: float,
                   novelty: float,
                   dopamine: float = 0.5) -> bool:
        """
        Decide if region should grow new connections.
        
        Grow if ALL:
        1. High prediction error (need more capacity)
        2. High novelty (new patterns to learn)
        3. Below max density (room to grow)
        4. Positive dopamine (rewarding to learn)
        """
        min_density, max_density = self.density_targets[region_type]
        
        return (
            prediction_error > self.growth_error_threshold and
            novelty > self.growth_novelty_threshold and
            current_density < max_density and
            dopamine > 0.5
        )
    
    def should_prune(self,
                    region_type: RegionType,
                    current_density: float,
                    redundancy: float) -> bool:
        """
        Decide if region should prune connections.
        
        Prune if:
        1. High redundancy OR
        2. Above max density
        """
        min_density, max_density = self.density_targets[region_type]
        
        return (
            redundancy > self.pruning_redundancy_threshold or
            current_density > max_density
        )
    
    def compute_growth_amount(self,
                             current_density: float,
                             target_density: float,
                             weights_shape: Tuple[int, int]) -> int:
        """Calculate how many connections to add."""
        density_gap = target_density - current_density
        total_possible = weights_shape[0] * weights_shape[1]
        
        return int(density_gap * total_possible * 0.1)  # Grow 10% of gap per step


class BrainLifecycleManager:
    """
    Manages full lifecycle: Active → Stabilize → Compress → Store → Reactivate
    """
    
    def __init__(self,
                 stabilization_threshold: float = 0.1,
                 compression_delay: int = 100):
        """
        Initialize lifecycle manager.
        
        Args:
            stabilization_threshold: Activity below this → stabilizing
            compression_delay: Episodes of low activity before compression
        """
        self.stabilization_threshold = stabilization_threshold
        self.compression_delay = compression_delay
        
        # Track region states
        self.region_phases = {}  # region_id -> LifecyclePhase
        self.region_activity_history = {}  # region_id -> [activity levels]
        self.episodes_since_active = {}  # region_id -> int
        
    def update_region_activity(self, region_id: str, activity: float):
        """Track activity for lifecycle management."""
        if region_id not in self.region_activity_history:
            self.region_activity_history[region_id] = []
            self.region_phases[region_id] = LifecyclePhase.ACTIVE
            self.episodes_since_active[region_id] = 0
        
        self.region_activity_history[region_id].append(activity)
        
        # Keep last 100 activity measurements
        if len(self.region_activity_history[region_id]) > 100:
            self.region_activity_history[region_id].pop(0)
        
        # Update phase
        self._update_lifecycle_phase(region_id, activity)
    
    def _update_lifecycle_phase(self, region_id: str, activity: float):
        """Update lifecycle phase based on activity."""
        current_phase = self.region_phases[region_id]
        
        if activity > self.stabilization_threshold:
            # High activity → active
            self.region_phases[region_id] = LifecyclePhase.ACTIVE
            self.episodes_since_active[region_id] = 0
            
        elif activity < self.stabilization_threshold:
            # Low activity
            self.episodes_since_active[region_id] += 1
            
            if self.episodes_since_active[region_id] < self.compression_delay // 2:
                self.region_phases[region_id] = LifecyclePhase.STABILIZING
            elif self.episodes_since_active[region_id] < self.compression_delay:
                # Ready to compress but not yet
                pass
            else:
                # Compress now
                if current_phase != LifecyclePhase.COMPRESSED:
                    self.region_phases[region_id] = LifecyclePhase.COMPRESSED
    
    def should_compress(self, region_id: str) -> bool:
        """Check if region should be compressed."""
        if region_id not in self.episodes_since_active:
            return False
        
        return self.episodes_since_active[region_id] >= self.compression_delay
    
    def should_decompress(self, region_id: str, needed_for_task: bool) -> bool:
        """Check if region should be decompressed."""
        phase = self.region_phases.get(region_id, LifecyclePhase.ACTIVE)
        
        return (
            phase in [LifecyclePhase.COMPRESSED, LifecyclePhase.STORED] and
            needed_for_task
        )
    
    def get_phase(self, region_id: str) -> LifecyclePhase:
        """Get current lifecycle phase for region."""
        return self.region_phases.get(region_id, LifecyclePhase.ACTIVE)
    
    def get_activity_trend(self, region_id: str, window: int = 10) -> float:
        """Get recent activity trend (increasing or decreasing)."""
        if region_id not in self.region_activity_history:
            return 0.0
        
        history = self.region_activity_history[region_id]
        if len(history) < window:
            return 0.0
        
        recent = history[-window:]
        older = history[-2*window:-window] if len(history) >= 2*window else history[:window]
        
        return np.mean(recent) - np.mean(older)


class PerformanceDrivenNeurogenesis:
    """
    Grows connections based on performance needs.
    
    Strategy: Add connections where errors are high and novelty is detected.
    """
    
    def __init__(self,
                 density_controller: AdaptiveDensityController,
                 base_growth_rate: float = 0.01):
        self.density_controller = density_controller
        self.base_growth_rate = base_growth_rate
        
        # Track growth history
        self.growth_events = []
    
    def grow_connections(self,
                        weights: np.ndarray,
                        region_type: RegionType,
                        error_map: np.ndarray,
                        novelty_score: float,
                        neuromodulators: Dict[str, float]) -> np.ndarray:
        """
        Add new connections based on errors and novelty.
        
        Args:
            weights: Current weight matrix
            region_type: Type of brain region
            error_map: Per-neuron prediction errors
            novelty_score: How novel is current situation
            neuromodulators: Current neuromodulator levels
            
        Returns:
            Updated weights with new connections
        """
        # Check if should grow
        current_density = self.density_controller.compute_current_density(weights)
        prediction_error = np.mean(error_map)
        
        should_grow = self.density_controller.should_grow(
            region_type,
            current_density,
            prediction_error,
            novelty_score,
            neuromodulators.get('dopamine', 0.5)
        )
        
        if not should_grow:
            return weights
        
        # Calculate growth amount
        min_density, max_density = self.density_controller.get_density_range(region_type)
        target_density = (min_density + max_density) / 2
        n_connections_to_add = self.density_controller.compute_growth_amount(
            current_density, target_density, weights.shape
        )
        
        # Grow connections where errors are highest
        new_weights = weights.copy()
        
        # Find neurons with highest errors
        error_sorted_indices = np.argsort(error_map)[::-1]
        high_error_neurons = error_sorted_indices[:len(error_sorted_indices)//10]  # Top 10%
        
        # Add connections from/to high-error neurons
        connections_added = 0
        for _ in range(n_connections_to_add):
            if connections_added >= n_connections_to_add:
                break
            
            # Random source from high-error neurons
            i = np.random.choice(high_error_neurons)
            j = np.random.randint(0, weights.shape[1])
            
            # Add if not already connected
            if new_weights[i, j] == 0:
                # Small initial weight, sign based on correlation
                new_weights[i, j] = np.random.randn() * 0.01
                connections_added += 1
        
        # Record growth event
        self.growth_events.append({
            'region_type': region_type,
            'connections_added': connections_added,
            'error': prediction_error,
            'novelty': novelty_score,
            'density_before': current_density,
            'density_after': self.density_controller.compute_current_density(new_weights)
        })
        
        return new_weights
    
    def get_growth_statistics(self) -> Dict:
        """Get statistics on neurogenesis events."""
        if not self.growth_events:
            return {}
        
        return {
            'total_growth_events': len(self.growth_events),
            'total_connections_added': sum(e['connections_added'] for e in self.growth_events),
            'average_error_at_growth': np.mean([e['error'] for e in self.growth_events]),
            'average_novelty_at_growth': np.mean([e['novelty'] for e in self.growth_events])
        }


class CompressionScheduler:
    """
    Decides when to compress regions based on lifecycle.
    """
    
    def __init__(self, lifecycle_manager: BrainLifecycleManager):
        self.lifecycle_manager = lifecycle_manager
        
        # Track compression stats
        self.compression_stats = {}
    
    def get_compression_priority(self, region_id: str) -> float:
        """
        Calculate compression priority (0-1).
        
        Higher priority = should compress sooner.
        """
        phase = self.lifecycle_manager.get_phase(region_id)
        
        if phase == LifecyclePhase.ACTIVE:
            return 0.0  # Don't compress active regions
        elif phase == LifecyclePhase.STABILIZING:
            return 0.3
        elif phase == LifecyclePhase.COMPRESSED:
            return 0.0  # Already compressed
        elif phase == LifecyclePhase.STORED:
            return 0.0  # Already stored
        
        # Check how long since active
        episodes = self.lifecycle_manager.episodes_since_active.get(region_id, 0)
        
        if episodes > 200:
            return 1.0  # Very high priority
        elif episodes > 100:
            return 0.7
        else:
            return 0.1
    
    def select_regions_to_compress(self,
                                   all_regions: List[str],
                                   target_compressed_fraction: float = 0.7) -> List[str]:
        """
        Select which regions to compress to meet target.
        
        Args:
            all_regions: List of all region IDs
            target_compressed_fraction: Target % of regions compressed
            
        Returns:
            List of region IDs to compress
        """
        # Get priorities
        priorities = {
            region_id: self.get_compression_priority(region_id)
            for region_id in all_regions
        }
        
        # Sort by priority
        sorted_regions = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
        
        # Select top regions until target met
        n_to_compress = int(len(all_regions) * target_compressed_fraction)
        
        # But only compress those with priority > 0
        to_compress = [
            region_id for region_id, priority in sorted_regions[:n_to_compress]
            if priority > 0
        ]
        
        return to_compress


def benchmark_adaptive_neurogenesis():
    """Benchmark adaptive neurogenesis system."""
    print("Benchmarking Adaptive Neurogenesis...")
    
    # Create components
    density_controller = AdaptiveDensityController()
    lifecycle_manager = BrainLifecycleManager()
    neurogenesis = PerformanceDrivenNeurogenesis(density_controller)
    
    # Simulate a brain region
    region_id = "hidden_layer_1"
    weights = np.random.randn(100, 100) * 0.1
    weights[np.random.random((100, 100)) < 0.95] = 0  # 95% sparse initially
    
    region_type = RegionType.HIDDEN
    
    print(f"\nInitial state:")
    print(f"  Density: {density_controller.compute_current_density(weights):.2%}")
    print(f"  Phase: {lifecycle_manager.get_phase(region_id).value}")
    
    # Simulate episodes
    n_episodes = 200
    
    for episode in range(n_episodes):
        # Simulate activity and errors
        activity = 0.5 + 0.3 * np.sin(episode / 20) + np.random.randn() * 0.1
        activity = np.clip(activity, 0, 1)
        
        error_map = np.random.rand(100) * activity
        novelty = max(0, 1.0 - episode / 100)  # Decreases over time
        
        neuromodulators = {
            'dopamine': 0.5 + 0.3 * np.random.randn(),
            'acetylcholine': novelty
        }
        
        # Update lifecycle
        lifecycle_manager.update_region_activity(region_id, activity)
        
        # Try to grow
        weights = neurogenesis.grow_connections(
            weights, region_type, error_map, novelty, neuromodulators
        )
        
        # Print occasional updates
        if episode % 50 == 0:
            density = density_controller.compute_current_density(weights)
            phase = lifecycle_manager.get_phase(region_id)
            print(f"\nEpisode {episode}:")
            print(f"  Density: {density:.2%}")
            print(f"  Phase: {phase.value}")
            print(f"  Activity: {activity:.2f}")
    
    # Final statistics
    stats = neurogenesis.get_growth_statistics()
    print(f"\nFinal statistics:")
    if stats:
        print(f"  Growth events: {stats.get('total_growth_events', 0)}")
        print(f"  Connections added: {stats.get('total_connections_added', 0)}")
        print(f"  Avg error at growth: {stats.get('average_error_at_growth', 0):.2f}")
    else:
        print(f"  No growth events occurred")
    print(f"  Final density: {density_controller.compute_current_density(weights):.2%}")
    
    return stats
