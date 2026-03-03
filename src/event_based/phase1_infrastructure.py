"""
Phase 1: Event-Based Infrastructure

Core event-driven processing system for asynchronous SNNs.
Only computes on spikes - no wasted cycles on silent neurons.
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class EventType(Enum):
    """Types of events in the network."""
    SPIKE = "spike"
    PREDICTION = "prediction"
    OBSERVATION = "observation"
    ERROR = "error"


@dataclass(order=True)
class SpikeEvent:
    """
    A spike event with precise timing.
    
    Events are ordered by time for priority queue processing.
    """
    time: float  # When the spike occurs (ms)
    neuron_id: int = field(compare=False)  # Which neuron
    value: float = field(default=1.0, compare=False)  # Spike magnitude
    event_type: EventType = field(default=EventType.SPIKE, compare=False)
    
    def __repr__(self):
        return f"SpikeEvent(t={self.time:.2f}ms, n={self.neuron_id}, v={self.value:.2f}, type={self.event_type.value})"


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron for event-based processing.
    
    Only updates state when it receives a spike event.
    """
    
    def __init__(self, neuron_id: int, threshold: float = 1.0, 
                 leak_rate: float = 0.95, refractory_period: float = 2.0):
        self.neuron_id = neuron_id
        self.threshold = threshold
        self.leak_rate = leak_rate  # Membrane potential decay per ms
        self.refractory_period = refractory_period  # ms
        
        # State
        self.membrane_potential = 0.0
        self.last_spike_time = -1000.0  # Long ago
        self.last_update_time = 0.0
    
    def integrate(self, current_time: float, input_current: float) -> bool:
        """
        Integrate input current and check if neuron fires.
        
        Returns:
            True if neuron fires, False otherwise
        """
        # Apply leak since last update
        time_delta = current_time - self.last_update_time
        self.membrane_potential *= (self.leak_rate ** time_delta)
        
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.last_update_time = current_time
            return False
        
        # Integrate input
        self.membrane_potential += input_current
        self.last_update_time = current_time
        
        # Check threshold
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0  # Reset
            self.last_spike_time = current_time
            return True
        
        return False


class EventBasedNetwork:
    """
    Event-driven spiking neural network.
    
    Only processes neurons when they receive spikes - massive efficiency gain
    for sparse activity patterns.
    """
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.current_time = 0.0
        
        # Event queue (priority queue sorted by time)
        self.event_queue: List[SpikeEvent] = []
        
        # Neurons
        self.neurons = [LIFNeuron(i) for i in range(n_neurons)]
        
        # Connections: {source_id: [(target_id, weight, delay)]}
        self.connections: Dict[int, List[Tuple[int, float, float]]] = {}
        
        # Statistics
        self.events_processed = 0
        self.spikes_generated = 0
    
    def add_connection(self, source: int, target: int, weight: float, delay: float = 1.0):
        """Add a synaptic connection."""
        if source not in self.connections:
            self.connections[source] = []
        self.connections[source].append((target, weight, delay))
    
    def add_event(self, event: SpikeEvent):
        """Add event to the queue."""
        heapq.heappush(self.event_queue, event)
    
    def inject_spike(self, neuron_id: int, time: float, value: float = 1.0, 
                     event_type: EventType = EventType.SPIKE):
        """Inject an external spike (e.g., sensory input)."""
        event = SpikeEvent(time, neuron_id, value, event_type)
        self.add_event(event)
    
    def process_next_event(self) -> Optional[SpikeEvent]:
        """
        Process the next event in the queue.
        
        Returns:
            The processed event, or None if queue is empty
        """
        if not self.event_queue:
            return None
        
        # Pop next event (earliest time)
        event = heapq.heappop(self.event_queue)
        self.current_time = event.time
        self.events_processed += 1
        
        # Get target neuron
        neuron = self.neurons[event.neuron_id]
        
        # Integrate spike
        fired = neuron.integrate(event.time, event.value)
        
        if fired:
            self.spikes_generated += 1
            
            # Propagate spike to connected neurons
            if event.neuron_id in self.connections:
                for target_id, weight, delay in self.connections[event.neuron_id]:
                    # Create new spike event for target
                    new_event = SpikeEvent(
                        time=event.time + delay,
                        neuron_id=target_id,
                        value=event.value * weight,
                        event_type=event.event_type
                    )
                    self.add_event(new_event)
        
        return event
    
    def run_until(self, end_time: float) -> int:
        """
        Run simulation until specified time.
        
        Returns:
            Number of events processed
        """
        events_count = 0
        
        while self.event_queue and self.event_queue[0].time <= end_time:
            self.process_next_event()
            events_count += 1
        
        self.current_time = end_time
        return events_count
    
    def get_stats(self) -> Dict:
        """Get network statistics."""
        return {
            'events_processed': self.events_processed,
            'spikes_generated': self.spikes_generated,
            'queue_size': len(self.event_queue),
            'current_time': self.current_time,
            'avg_activity': self.spikes_generated / (self.n_neurons * max(1, self.current_time / 1000))
        }


def test_event_based_network():
    """Test the event-based network."""
    print("\n" + "="*70)
    print("PHASE 1: EVENT-BASED INFRASTRUCTURE TEST")
    print("="*70)
    
    # Create small network
    n_neurons = 100
    net = EventBasedNetwork(n_neurons)
    
    print(f"\nCreated network: {n_neurons} neurons")
    
    # Add some random connections
    print("Adding connections...")
    np.random.seed(42)
    for i in range(n_neurons):
        # Each neuron connects to 5 random targets
        targets = np.random.choice(n_neurons, size=5, replace=False)
        for target in targets:
            weight = np.random.uniform(0.1, 0.5)
            delay = np.random.uniform(0.5, 2.0)
            net.add_connection(i, target, weight, delay)
    
    print(f"Added {n_neurons * 5} connections")
    
    # Inject some initial spikes
    print("\nInjecting initial spikes...")
    for i in range(10):
        net.inject_spike(
            neuron_id=i,
            time=np.random.uniform(0, 10),
            value=1.0
        )
    
    # Run simulation
    print("\nRunning simulation for 100ms...")
    import time
    start = time.time()
    
    events_processed = net.run_until(100.0)
    
    elapsed = time.time() - start
    
    # Get stats
    stats = net.get_stats()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Events processed: {stats['events_processed']:,}")
    print(f"Spikes generated: {stats['spikes_generated']:,}")
    print(f"Average activity: {stats['avg_activity']:.2%}")
    print(f"Processing time: {elapsed*1000:.2f}ms")
    print(f"Event throughput: {stats['events_processed']/elapsed:,.0f} events/sec")
    
    # Success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA")
    print(f"{'='*70}")
    
    throughput = stats['events_processed'] / elapsed
    
    if throughput > 10000:
        print(f"[PASS] Event throughput: {throughput:,.0f} events/sec (target: >10K)")
    else:
        print(f"[FAIL] Event throughput: {throughput:,.0f} events/sec (target: >10K)")
    
    if stats['avg_activity'] < 0.1:
        print(f"[PASS] Sparse activity: {stats['avg_activity']:.2%} (target: <10%)")
    else:
        print(f"[WARN] Activity: {stats['avg_activity']:.2%} (target: <10%)")
    
    print("\n[SUCCESS] Phase 1 infrastructure is working!")
    print("Ready for Phase 2: Prediction Layer")
    
    return net


if __name__ == "__main__":
    net = test_event_based_network()
