"""
Optimization Recommender

Analyzes benchmark results and provides actionable recommendations
to close gaps with biological systems.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import List, Dict, Any
from src.benchmarks.biological import BenchmarkResult


class OptimizationRecommendation:
    """Single optimization recommendation."""
    
    def __init__(self, priority: str, issue: str, suggestions: List[str]):
        self.priority = priority  # HIGH, MEDIUM, LOW
        self.issue = issue
        self.suggestions = suggestions
    
    def __repr__(self):
        return f"[{self.priority}] {self.issue}"


class OptimizationRecommender:
    """
    Analyzes benchmark gaps and recommends optimizations.
    """
    
    def analyze(self, results: List[BenchmarkResult]) -> List[OptimizationRecommendation]:
        """
        Analyze benchmark results and generate recommendations.
        
        Args:
            results: List of benchmark results
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Aggregate metrics
        avg_neuron_eff = sum(r.neuron_efficiency for r in results) / len(results)
        avg_learning_eff = sum(r.learning_efficiency for r in results) / len(results)
        avg_energy_eff = sum(r.energy_efficiency for r in results) / len(results)
        
        # Check neuron efficiency
        if avg_neuron_eff < 0.5:
            recommendations.append(OptimizationRecommendation(
                priority="HIGH",
                issue=f"Using {1/avg_neuron_eff:.1f}x more neurons than biological baseline",
                suggestions=[
                    "Increase Nash pruning aggressiveness (lower threshold from 0.05 to 0.02)",
                    "Start with sparse initialization (5% connectivity instead of random)",
                    "Use Phase 3.5 sparse matrices from the start",
                    "Implement connection death (remove unused connections)",
                    "Target: Match C. elegans 302 neurons for simple tasks"
                ]
            ))
        elif avg_neuron_eff < 0.8:
            recommendations.append(OptimizationRecommendation(
                priority="MEDIUM",
                issue=f"Using {1/avg_neuron_eff:.1f}x more neurons than biological",
                suggestions=[
                    "Fine-tune pruning threshold",
                    "Increase compression ratio",
                    "Better weight initialization"
                ]
            ))
        
        # Check learning efficiency
        if avg_learning_eff < 0.1:
            recommendations.append(OptimizationRecommendation(
                priority="HIGH",
                issue=f"Learning {1/avg_learning_eff:.0f}x slower than biological baseline",
                suggestions=[
                    "Implement eligibility traces (better credit assignment)",
                    "Use Phase 3e meta-learning to optimize learning rate",
                    "Add reward shaping (intermediate rewards)",
                    "Implement one-shot learning mechanisms",
                    "Better neuromodulator tuning (dopamine, serotonin)",
                    "Target: Match C. elegans 10-trial learning"
                ]
            ))
        elif avg_learning_eff < 0.5:
            recommendations.append(OptimizationRecommendation(
                priority="MEDIUM",
                issue=f"Learning {1/avg_learning_eff:.1f}x slower than biological",
                suggestions=[
                    "Increase learning rate",
                    "Better exploration strategy",
                    "Curriculum learning (start easier)"
                ]
            ))
        
        # Check energy efficiency
        if avg_energy_eff < 0.001:
            recommendations.append(OptimizationRecommendation(
                priority="MEDIUM",
                issue=f"Energy consumption {1/avg_energy_eff:.0f}x higher than biological",
                suggestions=[
                    "Deploy on neuromorphic hardware (Intel Loihi, BrainChip Akida)",
                    "Implement event-driven computation (only compute when needed)",
                    "Use spiking neural networks (SNNs) instead of continuous",
                    "Quantize weights to 4-8 bits (Phase 2 compression)",
                    "Note: This is hardware-limited, not algorithmic"
                ]
            ))
        
        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda r: priority_order[r.priority])
        
        return recommendations
    
    def print_recommendations(self, recommendations: List[OptimizationRecommendation]):
        """Print recommendations in readable format."""
        print("\n" + "="*70)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*70)
        
        if not recommendations:
            print("\n✓ No major optimizations needed - performing at biological levels!")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec.priority}] {rec.issue}")
            print("-" * 70)
            print("Suggestions:")
            for j, suggestion in enumerate(rec.suggestions, 1):
                print(f"  {j}. {suggestion}")
    
    def generate_action_plan(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Generate markdown action plan."""
        if not recommendations:
            return "# Action Plan\n\n✓ System performing at biological levels!\n"
        
        md = "# Optimization Action Plan\n\n"
        md += "Based on biological benchmarking results.\n\n"
        md += "---\n\n"
        
        high_priority = [r for r in recommendations if r.priority == "HIGH"]
        medium_priority = [r for r in recommendations if r.priority == "MEDIUM"]
        
        if high_priority:
            md += "## 🔴 High Priority (Address First)\n\n"
            for i, rec in enumerate(high_priority, 1):
                md += f"### {i}. {rec.issue}\n\n"
                md += "**Actions:**\n"
                for suggestion in rec.suggestions:
                    md += f"- [ ] {suggestion}\n"
                md += "\n"
        
        if medium_priority:
            md += "## 🟡 Medium Priority (Address After High)\n\n"
            for i, rec in enumerate(medium_priority, 1):
                md += f"### {i}. {rec.issue}\n\n"
                md += "**Actions:**\n"
                for suggestion in rec.suggestions:
                    md += f"- [ ] {suggestion}\n"
                md += "\n"
        
        md += "---\n\n"
        md += "## Next Steps\n\n"
        md += "1. Implement high-priority optimizations\n"
        md += "2. Re-run benchmarks\n"
        md += "3. Measure improvement\n"
        md += "4. Iterate until biological parity achieved\n"
        
        return md


def test_recommender():
    """Test optimization recommender."""
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDER TEST")
    print("="*70)
    
    # Create mock benchmark results
    from src.benchmarks.biological import BiologicalTier
    
    result1 = BenchmarkResult("Navigation", BiologicalTier.C_ELEGANS)
    result1.our_neurons = 500
    result1.our_trials = 200
    result1.our_energy = 1e-6
    result1.compute_efficiency()
    
    result2 = BenchmarkResult("Learning", BiologicalTier.C_ELEGANS)
    result2.our_neurons = 600
    result2.our_trials = 150
    result2.our_energy = 1e-6
    result2.compute_efficiency()
    
    results = [result1, result2]
    
    # Analyze
    recommender = OptimizationRecommender()
    recommendations = recommender.analyze(results)
    
    # Print
    recommender.print_recommendations(recommendations)
    
    # Generate action plan
    print("\n" + "="*70)
    print("GENERATING ACTION PLAN")
    print("="*70)
    
    action_plan = recommender.generate_action_plan(recommendations)
    print("\n" + action_plan)
    
    # Save to file
    with open('optimization_action_plan.md', 'w') as f:
        f.write(action_plan)
    
    print("✓ Saved action plan to 'optimization_action_plan.md'")
    
    return recommendations


if __name__ == "__main__":
    recommendations = test_recommender()
