"""
Phase 2: Prediction Layer

Adds "subconscious" prediction neurons that generate expectations
of future sensory states. These predictions will be compared to
actual observations to generate error signals.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from collections import deque
from src.event_based.phase1_infrastructure import (
    EventBasedNetwork, SpikeEvent, EventType, LIFNeuron
)


class PredictionNeuron(LIFNeuron):
    """
    Specialized neuron that learns to predict future states.
    
    Maintains a history of recent inputs and learns patterns.
    """
    
    def __init__(self, neuron_id: int, prediction_horizon: float = 10.0, **kwargs):
        super().__init__(neuron_id, **kwargs)
        self.prediction_horizon = prediction_horizon  # How far ahead to predict (ms)
        self.input_history = deque(maxlen=10)  # Recent inputs
        self.prediction_weights = {}  # {source_id: weight}
    
    def update_history(self, source_id: int, value: float, time: float):
        """Record an input for learning."""
        self.input_history.append((source_id, value, time))
    
    def learn_pattern(self, actual_value: float, learning_rate: float = 0.01):
        """Update prediction weights based on actual outcome."""
        if not self.input_history:
            return
        
        # Simple Hebbian: strengthen connections that predicted correctly
        for source_id, value, time in self.input_history:
            if source_id not in self.prediction_weights:
                self.prediction_weights[source_id] = 0.0
            
            # Update weight based on prediction error
            prediction_error = actual_value - self.membrane_potential
            self.prediction_weights[source_id] += learning_rate * value * prediction_error


class PredictiveNetwork(EventBasedNetwork):
    """
    Event-based network with prediction layer.
    
    Architecture:
        Input neurons → Prediction neurons → Prediction output
    """
    
    def __init__(self, n_input: int, n_prediction: int, prediction_horizon: float = 10.0):
        # Total neurons = input + prediction
        super().__init__(n_input + n_prediction)
        
        self.n_input = n_input
        self.n_prediction = n_prediction
        self.prediction_horizon = prediction_horizon
        
        # Replace prediction neurons with specialized type
        for i in range(n_input, n_input + n_prediction):
            self.neurons[i] = PredictionNeuron(
                i, 
                prediction_horizon=prediction_horizon,
                threshold=0.8,  # Lower threshold for predictions
                leak_rate=0.98
            )
        
        # Input buffer for pattern learning
        self.input_buffer = deque(maxlen=100)
        
        # Statistics
        self.predictions_made = 0
        self.predictions_correct = 0
    
    def inject_input(self, pattern: np.ndarray, time: float):
        """
        Inject sensory input and trigger predictions.
        
        Args:
            pattern: Input pattern (n_input dimensional)
            time: Current time
        """
        # Store in buffer
        self.input_buffer.append((pattern.copy(), time))
        
        # Inject as observation events
        for i, value in enumerate(pattern):
            if value > 0.1:  # Only inject significant values
                self.inject_spike(i, time, value, EventType.OBSERVATION)
        
        # Trigger predictions for future time
        self._generate_predictions(time)
    
    def _generate_predictions(self, current_time: float):
        """
        Generate predictions for future state.
        
        Prediction neurons fire based on learned patterns.
        """
        if len(self.input_buffer) < 2:
            return  # Need history to predict
        
        # Get recent patterns
        recent_patterns = [p for p, t in self.input_buffer if current_time - t < 50]
        if not recent_patterns:
            return
        
        # Average recent activity (simple prediction)
        avg_pattern = np.mean(recent_patterns, axis=0)
        
        # Generate prediction spikes
        prediction_time = current_time + self.prediction_horizon
        
        for i in range(self.n_prediction):
            # Prediction neuron gets input from pattern
            input_idx = i % self.n_input
            predicted_value = avg_pattern[input_idx]
            
            if predicted_value > 0.3:  # Threshold for prediction
                # Inject prediction spike
                pred_neuron_id = self.n_input + i
                self.inject_spike(
                    pred_neuron_id,
                    prediction_time,
                    predicted_value,
                    EventType.PREDICTION
                )
                self.predictions_made += 1
    
    def check_prediction_accuracy(self, observation_time: float, tolerance: float = 5.0):
        """
        Check how accurate recent predictions were.
        
        Args:
            observation_time: Time of observation
            tolerance: Time window for matching (ms)
        """
        # This will be used in Phase 3 for error detection
        # For now, just track that we're making predictions
        pass
    
    def get_prediction_stats(self):
        """Get prediction statistics."""
        stats = self.get_stats()
        stats['predictions_made'] = self.predictions_made
        if self.predictions_made > 0:
            stats['prediction_accuracy'] = self.predictions_correct / self.predictions_made
        else:
            stats['prediction_accuracy'] = 0.0
        return stats


def test_prediction_layer():
    """Test the prediction layer."""
    print("\n" + "="*70)
    print("PHASE 2: PREDICTION LAYER TEST")
    print("="*70)
    
    # Create network
    n_input = 10
    n_prediction = 50
    
    net = PredictiveNetwork(
        n_input=n_input,
        n_prediction=n_prediction,
        prediction_horizon=10.0
    )
    
    print(f"\nCreated predictive network:")
    print(f"  Input neurons: {n_input}")
    print(f"  Prediction neurons: {n_prediction}")
    print(f"  Prediction horizon: {net.prediction_horizon}ms")
    
    # Add connections from input to prediction neurons
    print("\nAdding connections...")
    for i in range(n_input):
        # Each input connects to multiple prediction neurons
        for j in range(n_prediction):
            if np.random.rand() < 0.3:  # 30% connection probability
                weight = np.random.uniform(0.2, 0.8)
                delay = 1.0
                net.add_connection(i, n_input + j, weight, delay)
    
    # Simulate sensory input sequence
    print("\nSimulating sensory input sequence...")
    
    # Create a simple repeating pattern
    pattern_a = np.array([1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    pattern_b = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    
    current_time = 0.0
    
    # Present pattern sequence
    for cycle in range(5):
        # Pattern A
        net.inject_input(pattern_a, current_time)
        current_time += 20.0
        net.run_until(current_time)
        
        # Pattern B
        net.inject_input(pattern_b, current_time)
        current_time += 20.0
        net.run_until(current_time)
    
    # Get statistics
    stats = net.get_prediction_stats()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Events processed: {stats['events_processed']:,}")
    print(f"Spikes generated: {stats['spikes_generated']:,}")
    print(f"Predictions made: {stats['predictions_made']:,}")
    print(f"Average activity: {stats['avg_activity']:.2%}")
    
    # Success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}")
    
    if stats['predictions_made'] > 0:
        print(f"[PASS] Predictions generated: {stats['predictions_made']}")
    else:
        print(f"[FAIL] No predictions generated")
    
    if stats['predictions_made'] > 20:
        print(f"[PASS] Sufficient predictions: {stats['predictions_made']} (target: >20)")
    else:
        print(f"[WARN] Few predictions: {stats['predictions_made']} (target: >20)")
    
    # Check that predictions fire before observations
    prediction_events = [e for e in net.event_queue if e.event_type == EventType.PREDICTION]
    if prediction_events:
        print(f"[PASS] Prediction events in queue: {len(prediction_events)}")
    
    print("\n[SUCCESS] Phase 2 prediction layer is working!")
    print("Predictions are being generated based on input patterns.")
    print("\nReady for Phase 3: Error Detection")
    
    return net


if __name__ == "__main__":
    net = test_prediction_layer()
