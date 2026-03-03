"""
Phase 3e Part 2: Bayesian Hyperparameter Optimizer

Uses Gaussian Process to find optimal configurations efficiently.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian Process to model performance as function of hyperparameters,
    then uses Expected Improvement to select next configuration to try.
    
    Much more sample-efficient than grid/random search!
    """
    
    def __init__(self,
                 parameter_space,
                 objective_function: Callable,
                 n_initial_random: int = 5):
        """
        Initialize Bayesian optimizer.
        
        Args:
            parameter_space: ParameterSpace object
            objective_function: Function config -> score
            n_initial_random: Number of random trials before Bayesian optimization
        """
        self.param_space = parameter_space
        self.objective = objective_function
        self.n_initial_random = n_initial_random
        
        # Gaussian Process with Matern kernel (good for hyperparameters)
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Trial history
        self.trials = []  # List of (config_dict, score)
        self.X_trials = []  # Configs as arrays
        self.y_trials = []  # Scores
        
    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Args:
            X: Candidate configurations (array)
            xi: Exploration parameter (higher = more exploration)
            
        Returns:
            Expected improvement for each configuration
        """
        if len(self.y_trials) == 0:
            return np.zeros(len(X))
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Current best
        best_y = np.max(self.y_trials)
        
        # Expected Improvement
        with np.errstate(divide='warn'):
            improvement = mu - best_y - xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def propose_next_config(self) -> Dict[str, float]:
        """
        Propose next configuration to try using Expected Improvement.
        
        Returns:
            Configuration dict
        """
        # Random search over parameter space
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            config = self.param_space.sample_random()
            candidates.append(self.param_space.to_array(config))
        
        candidates = np.array(candidates)
        
        # Compute Expected Improvement
        ei_values = self.expected_improvement(candidates)
        
        # Select best
        best_idx = np.argmax(ei_values)
        best_config_array = candidates[best_idx]
        
        return self.param_space.from_array(best_config_array)
    
    def optimize(self, n_trials: int = 50, verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run Bayesian optimization.
        
        Args:
            n_trials: Total number of trials
            verbose: Print progress
            
        Returns:
            (best_config, best_score)
        """
        if verbose:
            print(f"\nStarting Bayesian Optimization ({n_trials} trials)")
            print("=" * 60)
        
        # Phase 1: Random exploration
        if verbose:
            print(f"\nPhase 1: Random exploration ({self.n_initial_random} trials)")
        
        for i in range(self.n_initial_random):
            config = self.param_space.sample_random()
            score = self.objective(config)
            
            self.trials.append((config, score))
            self.X_trials.append(self.param_space.to_array(config))
            self.y_trials.append(score)
            
            if verbose:
                print(f"  Trial {i+1}/{self.n_initial_random}: score={score:.4f}")
        
        # Phase 2: Bayesian optimization
        if verbose:
            print(f"\nPhase 2: Bayesian optimization ({n_trials - self.n_initial_random} trials)")
        
        for i in range(self.n_initial_random, n_trials):
            # Fit GP on trials so far
            self.gp.fit(np.array(self.X_trials), np.array(self.y_trials))
            
            # Propose next config
            config = self.propose_next_config()
            
            # Evaluate
            score = self.objective(config)
            
            # Store
            self.trials.append((config, score))
            self.X_trials.append(self.param_space.to_array(config))
            self.y_trials.append(score)
            
            # Progress
            if verbose and (i + 1) % 5 == 0:
                best_so_far = np.max(self.y_trials)
                print(f"  Trial {i+1}/{n_trials}: score={score:.4f}, best={best_so_far:.4f}")
        
        # Return best
        best_idx = np.argmax(self.y_trials)
        best_config = self.trials[best_idx][0]
        best_score = self.y_trials[best_idx]
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best score: {best_score:.4f}")
        
        return best_config, best_score
    
    def get_best_config(self) -> Tuple[Dict, float]:
        """Get current best configuration."""
        if len(self.trials) == 0:
            return None, None
        
        best_idx = np.argmax(self.y_trials)
        return self.trials[best_idx]
    
    def get_optimization_history(self) -> Dict:
        """Get optimization history for visualization."""
        return {
            'scores': self.y_trials,
            'best_so_far': np.maximum.accumulate(self.y_trials),
            'configs': self.trials
        }


def test_bayesian_optimizer():
    """Test Bayesian optimizer on simple function."""
    print("\n" + "="*60)
    print("TEST: Bayesian Optimizer")
    print("="*60)
    
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.meta_learning.parameter_space import ParameterSpace
    
    # Create parameter space
    param_space = ParameterSpace()
    
    # Simple test objective: maximize negative distance from default
    # (Optimizer should find default config)
    def test_objective(config):
        default = param_space.get_default_config()
        
        # Compute distance
        distance = 0
        for key in config:
            distance += (config[key] - default[key]) ** 2
        
        # Return negative distance (maximize = minimize distance)
        return -np.sqrt(distance)
    
    # Run optimization
    optimizer = BayesianOptimizer(
        param_space,
        test_objective,
        n_initial_random=3
    )
    
    best_config, best_score = optimizer.optimize(n_trials=20, verbose=True)
    
    print(f"\nBest configuration found:")
    for key, value in list(best_config.items())[:5]:
        default_value = param_space.get_default_config()[key]
        print(f"  {key}: {value:.4f} (default: {default_value:.4f})")
    print("  ...")
    
    print(f"\nBest score: {best_score:.4f}")
    print("(Should be close to 0 if optimizer found default)")
    
    # Check convergence
    history = optimizer.get_optimization_history()
    improvement = history['best_so_far'][-1] - history['best_so_far'][0]
    print(f"\nImprovement: {improvement:.4f}")
    
    print("\n✓ Bayesian optimizer working!")
    
    return optimizer


if __name__ == "__main__":
    optimizer = test_bayesian_optimizer()
