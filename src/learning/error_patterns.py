"""
Phase 3d Part 1: Error Pattern Learning

Learn which connection patterns lead to errors.
Predict fragile connections before they fail.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import pickle


class ConnectionErrorLearner:
    """
    Learn which connections are error-prone.
    
    Tracks:
    - Connection strength over time
    - Usage patterns (how often activated)
    - Error correlations
    - Context (when errors occur)
    
    Predicts:
    - Risk score per connection (0-1)
    - High risk = needs redundancy
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize error learner.
        
        Args:
            history_size: Max error events to store
        """
        self.history_size = history_size
        
        # Error history per connection
        self.connection_errors = {}  # (i, j) -> [error_events]
        
        # Usage tracking
        self.connection_usage = {}  # (i, j) -> usage_count
        self.connection_variance = {}  # (i, j) -> weight_variance
        
        # Risk model (learned weights)
        self.risk_weights = {
            'weakness': 0.3,      # Low weight strength
            'isolation': 0.3,     # No redundant paths
            'instability': 0.2,   # High variance
            'history': 0.2        # Past errors
        }
        
        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
    
    def track_connection_usage(self, 
                               connection_id: Tuple[int, int],
                               weight: float,
                               activated: bool):
        """Track how connection is used."""
        if connection_id not in self.connection_usage:
            self.connection_usage[connection_id] = 0
            self.connection_variance[connection_id] = []
        
        if activated:
            self.connection_usage[connection_id] += 1
        
        # Track weight variance
        self.connection_variance[connection_id].append(weight)
        if len(self.connection_variance[connection_id]) > 100:
            self.connection_variance[connection_id].pop(0)
    
    def record_error(self,
                    connection_id: Tuple[int, int],
                    error: float,
                    context: Optional[Dict] = None):
        """
        Record error event for connection.
        
        Args:
            connection_id: (source, target) neuron indices
            error: Error magnitude
            context: Optional context (state, etc.)
        """
        if connection_id not in self.connection_errors:
            self.connection_errors[connection_id] = deque(maxlen=self.history_size)
        
        event = {
            'error': error,
            'timestamp': len(self.connection_errors[connection_id]),
            'context': context
        }
        
        self.connection_errors[connection_id].append(event)
    
    def analyze_connection_risk(self,
                                connection_id: Tuple[int, int],
                                weight: float,
                                redundancy_count: int) -> float:
        """
        Analyze risk score for connection.
        
        Risk factors:
        1. Weakness: Low weight strength
        2. Isolation: No redundant paths
        3. Instability: High weight variance
        4. History: Past errors
        
        Returns:
            Risk score (0-1), higher = more risky
        """
        risk = 0.0
        
        # Factor 1: Weakness
        weakness = max(0, 1.0 - abs(weight))  # Weak if close to 0
        risk += self.risk_weights['weakness'] * weakness
        
        # Factor 2: Isolation (no redundancy)
        isolation = 1.0 if redundancy_count == 0 else max(0, 1.0 - redundancy_count / 3)
        risk += self.risk_weights['isolation'] * isolation
        
        # Factor 3: Instability
        if connection_id in self.connection_variance:
            variance_history = self.connection_variance[connection_id]
            if len(variance_history) > 1:
                instability = np.std(variance_history) / (np.mean(np.abs(variance_history)) + 1e-10)
                instability = min(1.0, instability)  # Cap at 1.0
                risk += self.risk_weights['instability'] * instability
        
        # Factor 4: Error history
        if connection_id in self.connection_errors:
            error_history = self.connection_errors[connection_id]
            if len(error_history) > 0:
                recent_errors = [e['error'] for e in list(error_history)[-10:]]
                avg_error = np.mean(recent_errors)
                history_risk = min(1.0, avg_error)
                risk += self.risk_weights['history'] * history_risk
        
        return min(1.0, risk)  # Cap at 1.0
    
    def predict_all_risks(self,
                         weights: np.ndarray,
                         redundancy_map: Optional[Dict] = None) -> Dict[Tuple[int, int], float]:
        """
        Predict risk for all connections.
        
        Args:
            weights: Weight matrix
            redundancy_map: {(i,j): redundancy_count}
            
        Returns:
            {(i, j): risk_score}
        """
        risks = {}
        
        # Analyze all non-zero connections
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if weights[i, j] != 0:
                    connection_id = (i, j)
                    
                    # Get redundancy count
                    redundancy = 0
                    if redundancy_map and connection_id in redundancy_map:
                        redundancy = redundancy_map[connection_id]
                    
                    # Analyze risk
                    risk = self.analyze_connection_risk(
                        connection_id,
                        weights[i, j],
                        redundancy
                    )
                    
                    risks[connection_id] = risk
        
        return risks
    
    def learn_from_outcome(self,
                          connection_id: Tuple[int, int],
                          predicted_risk: float,
                          actual_error: float):
        """
        Update risk model based on prediction vs reality.
        
        If predicted high risk and high error → Good prediction
        If predicted low risk but high error → Bad prediction
        """
        self.total_predictions += 1
        
        # Threshold for "high"
        risk_threshold = 0.6
        error_threshold = 0.5
        
        predicted_high = predicted_risk > risk_threshold
        actual_high = actual_error > error_threshold
        
        if predicted_high == actual_high:
            self.correct_predictions += 1
            
            # Reinforce current weights (gradient ascent)
            # If correctly predicted high risk, slightly increase influence
            if predicted_high:
                # Identify which factor contributed most
                # (simplified: assume equal contribution for now)
                pass
        else:
            # Adjust weights (gradient descent)
            # If incorrectly predicted, adjust
            learning_rate = 0.01
            
            if predicted_high and not actual_high:
                # False positive - decrease weights
                for key in self.risk_weights:
                    self.risk_weights[key] *= (1 - learning_rate)
            elif not predicted_high and actual_high:
                # False negative - increase weights
                for key in self.risk_weights:
                    self.risk_weights[key] *= (1 + learning_rate)
            
            # Normalize
            total = sum(self.risk_weights.values())
            for key in self.risk_weights:
                self.risk_weights[key] /= total
    
    def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def get_high_risk_connections(self,
                                  risks: Dict[Tuple[int, int], float],
                                  threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Get list of high-risk connections.
        
        Args:
            risks: Risk scores
            threshold: Risk threshold
            
        Returns:
            List of connection IDs above threshold
        """
        return [conn_id for conn_id, risk in risks.items() if risk > threshold]
    
    def compress_history(self, compression_engine=None):
        """
        Compress error history using Phase 3c statistical sampling.
        
        This prevents unbounded growth of error history.
        """
        if compression_engine is None:
            # Simple compression: keep only recent + high-error events
            for connection_id in self.connection_errors:
                events = list(self.connection_errors[connection_id])
                
                if len(events) > self.history_size:
                    # Sort by error magnitude
                    events.sorted(key=lambda e: e['error'], reverse=True)
                    
                    # Keep top 50% by error + most recent 50%
                    high_error = events[:len(events)//2]
                    recent = events[-len(events)//2:]
                    
                    # Combine and deduplicate
                    kept = list(set(high_error + recent))
                    
                    self.connection_errors[connection_id] = deque(kept, maxlen=self.history_size)
        else:
            # Use Phase 3c compression (future integration)
            pass
    
    def get_statistics(self) -> Dict:
        """Get learner statistics."""
        return {
            'total_connections_tracked': len(self.connection_usage),
            'total_connections_with_errors': len(self.connection_errors),
            'total_predictions': self.total_predictions,
            'prediction_accuracy': self.get_prediction_accuracy(),
            'risk_model_weights': self.risk_weights.copy()
        }


def benchmark_error_learning():
    """Benchmark error pattern learning."""
    print("\nBenchmarking Error Pattern Learning...")
    
    # Create test network
    n = 100
    weights = np.random.randn(n, n) * 0.1
    weights[np.random.random((n, n)) < 0.9] = 0  # 90% sparse
    
    learner = ConnectionErrorLearner()
    
    print(f"\nTest network: {n}×{n}")
    print(f"Non-zero connections: {np.count_nonzero(weights)}")
    
    # Simulate usage and errors
    print("\nSimulating 100 episodes...")
    
    for episode in range(100):
        # Track some connections
        for i in range(n):
            for j in range(n):
                if weights[i, j] != 0:
                    # Simulate activation (random for now)
                    activated = np.random.random() < 0.3
                    learner.track_connection_usage((i, j), weights[i, j], activated)
                    
                    # Simulate occasional errors (weak connections more likely)
                    if abs(weights[i, j]) < 0.05 and np.random.random() < 0.1:
                        error = np.random.random()
                        learner.record_error((i, j), error)
    
    # Predict risks
    risks = learner.predict_all_risks(weights)
    
    print(f"\nRisk analysis:")
    print(f"  Connections analyzed: {len(risks)}")
    
    # High risk connections
    high_risk = learner.get_high_risk_connections(risks, threshold=0.7)
    print(f"  High risk (>0.7): {len(high_risk)}")
    
    # Statistics
    stats = learner.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return learner
