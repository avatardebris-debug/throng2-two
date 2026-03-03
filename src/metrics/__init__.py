"""
Metrics module for information theory and signal-to-noise analysis.
"""

from .information_theory import (
    shannon_entropy,
    mutual_information,
    signal_to_noise_ratio,
    calculate_snr_shannon,
    calculate_snr_fourier
)

__all__ = [
    'shannon_entropy',
    'mutual_information',
    'signal_to_noise_ratio',
    'calculate_snr_shannon',
    'calculate_snr_fourier'
]
