"""
Phase 3e Part 1: Parameter Space Definition

Enumerate all tunable hyperparameters across Phases 3-3d.
"""

import numpy as np
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass


@dataclass
class Parameter:
    """Single hyperparameter definition."""
    name: str
    bounds: Tuple[float, float]
    param_type: str  # 'continuous' or 'integer'
    default: float
    description: str


class ParameterSpace:
    """
    Complete hyperparameter space for Thronglet Brain.
    
    Covers all tunable parameters from Phases 3-3d:
    - Nash pruning
    - Adaptive neurogenesis
    - Statistical sampling
    - Predictive error-reduction
    - Neuromodulators
    """
    
    def __init__(self):
        self.parameters = self._define_parameters()
        
    def _define_parameters(self) -> Dict[str, Parameter]:
        """Define all tunable parameters."""
        
        params = {}
        
        # ===== Phase 3: Nash Pruning =====
        params['nash_pruning_threshold'] = Parameter(
            name='nash_pruning_threshold',
            bounds=(0.01, 0.2),
            param_type='continuous',
            default=0.05,
            description='Threshold for pruning weak connections'
        )
        
        params['nash_frequency'] = Parameter(
            name='nash_frequency',
            bounds=(50, 200),
            param_type='integer',
            default=100,
            description='Episodes between pruning operations'
        )
        
        # ===== Phase 3b: Adaptive Neurogenesis =====
        # Density targets (min, max) per region
        params['sensory_density_min'] = Parameter(
            name='sensory_density_min',
            bounds=(0.05, 0.15),
            param_type='continuous',
            default=0.10,
            description='Minimum connection density for sensory regions'
        )
        
        params['sensory_density_max'] = Parameter(
            name='sensory_density_max',
            bounds=(0.15, 0.30),
            param_type='continuous',
            default=0.20,
            description='Maximum connection density for sensory regions'
        )
        
        params['hidden_density_min'] = Parameter(
            name='hidden_density_min',
            bounds=(0.02, 0.10),
            param_type='continuous',
            default=0.05,
            description='Minimum connection density for hidden regions'
        )
        
        params['hidden_density_max'] = Parameter(
            name='hidden_density_max',
            bounds=(0.05, 0.15),
            param_type='continuous',
            default=0.10,
            description='Maximum connection density for hidden regions'
        )
        
        params['output_density_min'] = Parameter(
            name='output_density_min',
            bounds=(0.10, 0.20),
            param_type='continuous',
            default=0.15,
            description='Minimum connection density for output regions'
        )
        
        params['output_density_max'] = Parameter(
            name='output_density_max',
            bounds=(0.20, 0.35),
            param_type='continuous',
            default=0.25,
            description='Maximum connection density for output regions'
        )
        
        params['growth_error_threshold'] = Parameter(
            name='growth_error_threshold',
            bounds=(0.3, 0.7),
            param_type='continuous',
            default=0.5,
            description='Error threshold for triggering growth'
        )
        
        params['growth_novelty_threshold'] = Parameter(
            name='growth_novelty_threshold',
            bounds=(0.5, 0.9),
            param_type='continuous',
            default=0.7,
            description='Novelty threshold for triggering growth'
        )
        
        # ===== Phase 3c: Statistical Sampling =====
        params['sample_fraction'] = Parameter(
            name='sample_fraction',
            bounds=(0.001, 0.05),
            param_type='continuous',
            default=0.01,
            description='Fraction of weights to sample'
        )
        
        params['bootstrap_samples'] = Parameter(
            name='bootstrap_samples',
            bounds=(3, 10),
            param_type='integer',
            default=5,
            description='Number of bootstrap reconstructions'
        )
        
        # ===== Phase 3d: Predictive Error-Reduction =====
        params['error_threshold'] = Parameter(
            name='error_threshold',
            bounds=(0.1, 0.6),
            param_type='continuous',
            default=0.3,
            description='Error threshold for enabling Phase 3d'
        )
        
        params['redundancy_threshold'] = Parameter(
            name='redundancy_threshold',
            bounds=(0.5, 0.9),
            param_type='continuous',
            default=0.7,
            description='Risk threshold for adding redundancy'
        )
        
        params['update_frequency'] = Parameter(
            name='update_frequency',
            bounds=(5, 20),
            param_type='integer',
            default=10,
            description='Episodes between Phase 3d updates'
        )
        
        # ===== Neuromodulators =====
        params['dopamine_baseline'] = Parameter(
            name='dopamine_baseline',
            bounds=(0.3, 0.7),
            param_type='continuous',
            default=0.5,
            description='Baseline dopamine level'
        )
        
        params['serotonin_baseline'] = Parameter(
            name='serotonin_baseline',
            bounds=(0.3, 0.7),
            param_type='continuous',
            default=0.5,
            description='Baseline serotonin level'
        )
        
        params['norepinephrine_baseline'] = Parameter(
            name='norepinephrine_baseline',
            bounds=(0.3, 0.7),
            param_type='continuous',
            default=0.5,
            description='Baseline norepinephrine level'
        )
        
        params['acetylcholine_baseline'] = Parameter(
            name='acetylcholine_baseline',
            bounds=(0.3, 0.7),
            param_type='continuous',
            default=0.5,
            description='Baseline acetylcholine level'
        )
        
        # ===== Learning Rates =====
        params['learning_rate'] = Parameter(
            name='learning_rate',
            bounds=(0.001, 0.1),
            param_type='continuous',
            default=0.01,
            description='Base learning rate'
        )
        
        params['discount_factor'] = Parameter(
            name='discount_factor',
            bounds=(0.9, 0.99),
            param_type='continuous',
            default=0.95,
            description='Temporal discount factor (gamma)'
        )
        
        return params
    
    def sample_random(self) -> Dict[str, float]:
        """Sample random configuration."""
        config = {}
        
        for name, param in self.parameters.items():
            low, high = param.bounds
            
            if param.param_type == 'continuous':
                config[name] = np.random.uniform(low, high)
            else:  # integer
                config[name] = np.random.randint(low, high + 1)
        
        return config
    
    def get_default_config(self) -> Dict[str, float]:
        """Get default (hand-tuned) configuration."""
        return {name: param.default for name, param in self.parameters.items()}
    
    def validate_config(self, config: Dict[str, float]) -> bool:
        """Check if configuration is valid."""
        for name, value in config.items():
            if name not in self.parameters:
                return False
            
            param = self.parameters[name]
            low, high = param.bounds
            
            if not (low <= value <= high):
                return False
            
            if param.param_type == 'integer' and not isinstance(value, int):
                return False
        
        return True
    
    def clip_config(self, config: Dict[str, float]) -> Dict[str, float]:
        """Clip configuration to valid bounds."""
        clipped = {}
        
        for name, value in config.items():
            param = self.parameters[name]
            low, high = param.bounds
            
            clipped_value = np.clip(value, low, high)
            
            if param.param_type == 'integer':
                clipped_value = int(round(clipped_value))
            
            clipped[name] = clipped_value
        
        return clipped
    
    def to_array(self, config: Dict[str, float]) -> np.ndarray:
        """Convert config dict to array (for GP)."""
        return np.array([config[name] for name in sorted(self.parameters.keys())])
    
    def from_array(self, array: np.ndarray) -> Dict[str, float]:
        """Convert array to config dict."""
        config = {}
        for i, name in enumerate(sorted(self.parameters.keys())):
            config[name] = array[i]
        
        return self.clip_config(config)
    
    def get_parameter_info(self) -> str:
        """Get human-readable parameter info."""
        info = "Parameter Space:\n"
        info += "=" * 60 + "\n\n"
        
        for name, param in sorted(self.parameters.items()):
            info += f"{name}:\n"
            info += f"  Range: {param.bounds}\n"
            info += f"  Type: {param.param_type}\n"
            info += f"  Default: {param.default}\n"
            info += f"  Description: {param.description}\n\n"
        
        return info
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return len(self.parameters)


def test_parameter_space():
    """Test parameter space functionality."""
    print("\n" + "="*60)
    print("TEST: Parameter Space")
    print("="*60)
    
    space = ParameterSpace()
    
    print(f"\nTotal parameters: {space.count_parameters()}")
    
    # Test default config
    default = space.get_default_config()
    print(f"\nDefault config has {len(default)} parameters")
    
    # Test random sampling
    random_config = space.sample_random()
    print(f"\nRandom config:")
    for key, value in list(random_config.items())[:5]:
        print(f"  {key}: {value:.4f}")
    print("  ...")
    
    # Test validation
    assert space.validate_config(default), "Default config should be valid"
    assert space.validate_config(random_config), "Random config should be valid"
    
    # Test array conversion
    array = space.to_array(default)
    recovered = space.from_array(array)
    assert all(abs(default[k] - recovered[k]) < 1e-6 for k in default), "Array conversion should be reversible"
    
    print("\n✓ Parameter space working!")
    
    return space


if __name__ == "__main__":
    space = test_parameter_space()
    
    # Print full info
    print("\n" + space.get_parameter_info())
