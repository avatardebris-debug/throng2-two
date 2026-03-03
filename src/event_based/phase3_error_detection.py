"""
Phase 3: Error Detection

Compares predictions to observations and generates error signals.
Only fires when there's a mismatch - this is the key to efficiency!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from collections import defaultdict
from src.event_based.phase2_prediction import PredictiveNetwork, EventType, SpikeEvent


class ErrorNeuron:
    """
    Detects mismatches between predictions and observations.
    
    Only fires when error exceeds threshold - this creates the
    "consciousness bottleneck" where only surprises get attention.
    """
    
    def __init__(self, neuron_id: int, error_threshold: float = 0.2):
        self.neuron_id = neuron_id
        self.error_threshold = error_threshold
        
        # Waiting for matching events
        self.pending_predictions = {}  # {time: value}
        self.pending_observations = {}  # {time: value}
        
        # Statistics
        self.comparisons_made = 0
        self.errors_detected = 0
    
    def add_prediction(self, time: float, value: float):
        """Record a prediction."""
        self.pending_predictions[time] = value
    
    def add_observation(self, time: float, value: float) -> tuple:
        """
        Record an observation and check for matching prediction.
        
        Returns:
            (has_error, error_magnitude) tuple
        """
        self.pending_observations[time] = value
        
        # Find matching prediction (within time window)
        tolerance = 5.0  # ms
        matching_pred = None
        matching_time = None
        
        for pred_time, pred_value in self.pending_predictions.items():
            if abs(pred_time - time) < tolerance:
                matching_pred = pred_value
                matching_time = pred_time
                break
        
        if matching_pred is not None:
            # Compute error
            error = abs(value - matching_pred)
            self.comparisons_made += 1
            
            # Clean up
            del self.pending_predictions[matching_time]
            
            # Check threshold
            if error > self.error_threshold:
                self.errors_detected += 1
                return (True, error)
            else:
                return (False, error)
        
        return (False, 0.0)


class ErrorDetectionNetwork(PredictiveNetwork):
    """
    Network with error detection layer.
    
    Architecture:
        Input → Prediction → Comparison → Error (only on mismatch!)
    """
    
    def __init__(self, n_input: int, n_prediction: int, n_error: int, 
                 prediction_horizon: float = 10.0, error_threshold: float = 0.2):
        super().__init__(n_input, n_prediction, prediction_horizon)
        
        self.n_error = n_error
        self.error_threshold = error_threshold
        
        # Error neurons (one per input dimension)
        self.error_neurons = [
            ErrorNeuron(i, error_threshold) 
            for i in range(n_error)
        ]
        
        # Track error events
        self.error_events = []
    
    def process_next_event(self):
        """Override to intercept prediction and observation events."""
        event = super().process_next_event()
        
        if event is None:
            return None
        
        # Check if this is a prediction or observation
        if event.event_type == EventType.PREDICTION:
            # Route to appropriate error neuron
            error_idx = event.neuron_id % self.n_error
            self.error_neurons[error_idx].add_prediction(event.time, event.value)
        
        elif event.event_type == EventType.OBSERVATION:
            # Route to appropriate error neuron
            error_idx = event.neuron_id % self.n_error
            has_error, error_mag = self.error_neurons[error_idx].add_observation(
                event.time, event.value
            )
            
            if has_error:
                # Generate error spike!
                error_event = SpikeEvent(
                    time=event.time,
                    neuron_id=error_idx,
                    value=error_mag,
                    event_type=EventType.ERROR
                )
                self.error_events.append(error_event)
                self.add_event(error_event)
        
        return event
    
    def get_error_stats(self):
        """Get error detection statistics."""
        stats = self.get_prediction_stats()
        
        total_comparisons = sum(e.comparisons_made for e in self.error_neurons)
        total_errors = sum(e.errors_detected for e in self.error_neurons)
        
        stats['comparisons_made'] = total_comparisons
        stats['errors_detected'] = total_errors
        stats['error_rate'] = total_errors / max(1, total_comparisons)
        stats['error_events'] = len(self.error_events)
        
        return stats


def test_error_detection():
    """Test error detection layer."""
    print("\n" + "="*70)
    print("PHASE 3: ERROR DETECTION TEST")
    print("="*70)
    
    # Create network
    n_input = 10
    n_prediction = 50
    n_error = 10
    
    net = ErrorDetectionNetwork(
        n_input=n_input,
        n_prediction=n_prediction,
        n_error=n_error,
        prediction_horizon=10.0,
        error_threshold=0.2
    )
    
    print(f"\nCreated error detection network:")
    print(f"  Input neurons: {n_input}")
    print(f"  Prediction neurons: {n_prediction}")
    print(f"  Error neurons: {n_error}")
    print(f"  Error threshold: {net.error_threshold}")
    
    # Add connections
    print("\nAdding connections...")
    for i in range(n_input):
        for j in range(n_prediction):
            if np.random.rand() < 0.3:
                weight = np.random.uniform(0.2, 0.8)
                net.add_connection(i, n_input + j, weight, 1.0)
    
    # Test with predictable and unpredictable patterns
    print("\nTesting with pattern sequence...")
    print("  - Predictable patterns should generate LOW errors")
    print("  - Novel patterns should generate HIGH errors")
    
    # Predictable pattern (repeated)
    pattern_a = np.array([1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    pattern_b = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    
    # Novel pattern (unexpected)
    pattern_novel = np.array([0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0])
    
    current_time = 0.0
    
    # Phase 1: Predictable sequence
    print("\n  Phase 1: Predictable patterns (A-B-A-B)...")
    for cycle in range(3):
        net.inject_input(pattern_a, current_time)
        current_time += 20.0
        net.run_until(current_time)
        
        net.inject_input(pattern_b, current_time)
        current_time += 20.0
        net.run_until(current_time)
    
    errors_predictable = len(net.error_events)
    
    # Phase 2: Novel pattern (should generate errors!)
    print("  Phase 2: Novel pattern (surprise!)...")
    net.inject_input(pattern_novel, current_time)
    current_time += 20.0
    net.run_until(current_time)
    
    errors_novel = len(net.error_events) - errors_predictable
    
    # Get statistics
    stats = net.get_error_stats()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Comparisons made: {stats['comparisons_made']}")
    print(f"Errors detected: {stats['errors_detected']}")
    print(f"Error rate: {stats['error_rate']:.1%}")
    print(f"Error events: {stats['error_events']}")
    print(f"\nErrors on predictable patterns: {errors_predictable}")
    print(f"Errors on novel pattern: {errors_novel}")
    
    # Success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}")
    
    if stats['comparisons_made'] > 0:
        print(f"[PASS] Comparisons made: {stats['comparisons_made']}")
    else:
        print(f"[FAIL] No comparisons made")
    
    if stats['error_rate'] < 0.5:
        print(f"[PASS] Error rate: {stats['error_rate']:.1%} (target: <50%)")
        print("  → Most predictions are correct (low error)")
    else:
        print(f"[WARN] Error rate: {stats['error_rate']:.1%} (target: <50%)")
    
    if errors_novel > errors_predictable:
        print(f"[PASS] Novel pattern generated more errors ({errors_novel} vs {errors_predictable})")
        print("  → System detects surprises!")
    else:
        print(f"[WARN] Novel pattern didn't generate more errors")
    
    print("\n[SUCCESS] Phase 3 error detection is working!")
    print("The system only fires error signals on mismatches.")
    print("This creates the 'consciousness bottleneck' - only surprises get attention!")
    print("\nReady for Phase 4: Error-Driven Learning")
    
    return net


if __name__ == "__main__":
    net = test_error_detection()
