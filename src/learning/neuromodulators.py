"""
Neuromodulator Systems - Dopamine, serotonin, norepinephrine, acetylcholine.

Each maps to specific computational role:
- Dopamine: Reward prediction error (TD learning signal)
- Serotonin: Time horizon / patience
- Norepinephrine: Learning rate / urgency
- Acetylcholine: Attention / novelty detection
"""

import numpy as np
from typing import Dict


class NeuromodulatorSystem:
    """
    Neurochemical reward and control systems.
    
    Modulates learning and behavior based on context.
    """
    
    def __init__(self,
                 dopamine_baseline: float = 0.5,
                 serotonin_baseline: float = 0.5,
                 norepinephrine_baseline: float = 0.5,
                 acetylcholine_baseline: float = 0.5):
        """
        Initialize neuromodulator systems.
        
        All levels are 0-1, with 0.5 as baseline.
        """
        self.dopamine = dopamine_baseline
        self.serotonin = serotonin_baseline
        self.norepinephrine = norepinephrine_baseline
        self.acetylcholine = acetylcholine_baseline
        
        # For TD learning
        self.value_estimates = {}  # State -> expected value
        self.gamma = 0.95  # Discount factor (affected by serotonin)
        
    def compute_td_error(self, 
                        state: tuple, 
                        reward: float, 
                        next_state: tuple,
                        done: bool = False) -> float:
        """
        Compute TD error (dopamine signal).
        
        TD Error = Reality - Expectation
        - Positive = better than expected (dopamine burst)
        - Negative = worse than expected (dopamine dip)
        
        Args:
            state: Current state
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            TD error (becomes dopamine level)
        """
        # Get value estimates
        current_value = self.value_estimates.get(state, 0.0)
        
        if done:
            next_value = 0.0
        else:
            next_value = self.value_estimates.get(next_state, 0.0)
            
        # TD error = reward + γ*V(next) - V(current)
        td_error = reward + self.gamma * next_value - current_value
        
        # Update value estimate
        learning_rate = 0.1
        self.value_estimates[state] = current_value + learning_rate * td_error
        
        # Convert to dopamine level (0-1)
        # Positive TD error -> high dopamine
        # Negative TD error -> low dopamine
        self.dopamine = 0.5 + 0.5 * np.tanh(td_error)
        
        return td_error
    
    def update_from_context(self, 
                           uncertainty: float = 0.5,
                           stakes: float = 0.5,
                           novelty: float = 0.5):
        """
        Update neuromodulator levels based on context.
        
        Args:
            uncertainty: How uncertain is the situation? (0-1)
            stakes: How important is this decision? (0-1)
            novelty: How novel/unexpected? (0-1)
        """
        # High uncertainty -> high norepinephrine (learn fast, be alert)
        self.norepinephrine = 0.3 + 0.7 * uncertainty
        
        # High stakes -> high serotonin (be patient, think long-term)
        self.serotonin = 0.3 + 0.7 * stakes
        
        # High novelty -> high acetylcholine (pay attention, encode strongly)
        self.acetylcholine = 0.3 + 0.7 * novelty
        
        # Update discount factor based on serotonin
        # High serotonin = patient = high gamma (value future)
        self.gamma = 0.8 + 0.15 * self.serotonin
        
    def modulate_learning_rate(self, base_rate: float = 0.01) -> float:
        """
        Get modulated learning rate.
        
        Norepinephrine increases learning rate (urgency).
        
        Args:
            base_rate: Base learning rate
            
        Returns:
            Modulated learning rate
        """
        return base_rate * (0.5 + 1.5 * self.norepinephrine)
    
    def modulate_hebbian(self, base_rate: float = 0.01) -> float:
        """
        Get Hebbian learning rate modulated by dopamine and norepinephrine.
        
        Args:
            base_rate: Base Hebbian rate
            
        Returns:
            Modulated rate
        """
        # Dopamine gates learning (only strengthen if rewarded)
        # Norepinephrine speeds it up
        return base_rate * self.dopamine * (0.5 + 1.5 * self.norepinephrine)
    
    def get_exploration_rate(self, base_exploration: float = 0.3) -> float:
        """
        Get exploration rate.
        
        High acetylcholine (novelty) -> more exploration
        High norepinephrine (uncertainty) -> more exploration
        
        Args:
            base_exploration: Base exploration probability
            
        Returns:
            Modulated exploration rate
        """
        # Increase exploration when novel or uncertain
        novelty_factor = 0.5 + 0.5 * self.acetylcholine
        uncertainty_factor = 0.5 + 0.5 * self.norepinephrine
        
        return base_exploration * novelty_factor * uncertainty_factor
    
    def should_use_system_2(self, 
                           uncertainty_threshold: float = 0.7,
                           stakes_threshold: float = 0.7) -> bool:
        """
        Decide whether to use System 2 (slow, deliberate) processing.
        
        System 1: Fast, pattern-matching, cheap (2-5 watts)
        System 2: Slow, deliberate simulation, expensive (15-20 watts)
        
        Args:
            uncertainty_threshold: Switch to System 2 if uncertainty > this
            stakes_threshold: Switch to System 2 if stakes > this
            
        Returns:
            True if should use System 2
        """
        # Use System 2 when uncertain or high stakes
        return (self.norepinephrine > uncertainty_threshold or 
                self.serotonin > stakes_threshold)
    
    def get_levels(self) -> Dict[str, float]:
        """Get current neuromodulator levels."""
        return {
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'norepinephrine': self.norepinephrine,
            'acetylcholine': self.acetylcholine,
            'gamma': self.gamma
        }
    
    def reset_to_baseline(self):
        """Reset to baseline levels."""
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.norepinephrine = 0.5
        self.acetylcholine = 0.5
        self.gamma = 0.95
