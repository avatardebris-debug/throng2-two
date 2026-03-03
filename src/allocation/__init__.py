"""
Allocation module for Kelly Criterion-based resource management.
"""

from .kelly_allocator import (
    KellyAllocator,
    calculate_kelly_fraction,
    estimate_win_probability,
    estimate_expected_payoff
)

__all__ = [
    'KellyAllocator',
    'calculate_kelly_fraction',
    'estimate_win_probability',
    'estimate_expected_payoff'
]
