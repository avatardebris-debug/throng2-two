"""
Information theory utilities for signal-to-noise analysis.

Implements Shannon entropy, mutual information, and SNR calculations
to identify which connections carry real information vs noise.
"""

import numpy as np
from typing import Optional, Tuple


def shannon_entropy(weights: np.ndarray, bins: int = 50) -> float:
    """
    Calculate Shannon entropy of weight distribution.
    
    High entropy = noisy/random weights
    Low entropy = structured/informative weights
    
    Args:
        weights: Weight matrix or vector
        bins: Number of bins for histogram
        
    Returns:
        Entropy in bits
    """
    # Flatten weights
    w_flat = weights.flatten()
    
    # Create histogram (probability distribution)
    hist, _ = np.histogram(w_flat, bins=bins, density=True)
    
    # Normalize to probabilities
    hist = hist / hist.sum()
    
    # Remove zeros (log(0) is undefined)
    hist = hist[hist > 0]
    
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Calculate mutual information between two neuron groups.
    
    MI measures how much knowing one variable reduces uncertainty about another.
    High MI = strong correlation (useful connection)
    Low MI = independent (potentially redundant connection)
    
    Args:
        x: Activity of first neuron group
        y: Activity of second neuron group
        bins: Number of bins for 2D histogram
        
    Returns:
        Mutual information in bits
    """
    # Create 2D histogram
    hist_2d, _, _ = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
    
    # Normalize to joint probability
    p_xy = hist_2d / hist_2d.sum()
    
    # Marginal probabilities
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # Remove zeros
    p_xy = p_xy[p_xy > 0]
    
    # Calculate MI: sum(p(x,y) * log2(p(x,y) / (p(x)*p(y))))
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy.size > 0 and p_x[i] > 0 and p_y[j] > 0:
                idx = i * len(p_y) + j
                if idx < len(p_xy):
                    mi += p_xy[idx] * np.log2(p_xy[idx] / (p_x[i] * p_y[j]))
    
    return mi


def signal_to_noise_ratio(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Args:
        signal: Signal component
        noise: Noise component
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_snr_shannon(weights: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate SNR using Shannon entropy approach.
    
    Signal = low-entropy (structured) weights
    Noise = high-entropy (random) weights
    
    Args:
        weights: Weight matrix
        threshold: Threshold to separate signal from noise
        
    Returns:
        SNR ratio (signal_entropy / total_entropy)
    """
    # Separate into strong (signal) and weak (noise) weights
    strong_weights = weights[np.abs(weights) > threshold]
    weak_weights = weights[np.abs(weights) <= threshold]
    
    if len(strong_weights) == 0 or len(weak_weights) == 0:
        return 1.0
    
    # Calculate entropies
    signal_entropy = shannon_entropy(strong_weights)
    noise_entropy = shannon_entropy(weak_weights)
    
    if noise_entropy == 0:
        return float('inf')
    
    # SNR = signal_entropy / noise_entropy (lower is better for signal)
    # Invert so higher is better
    snr = noise_entropy / (signal_entropy + 1e-10)
    
    return snr


def calculate_snr_fourier(weights: np.ndarray, 
                          low_freq_cutoff: float = 0.1) -> Tuple[float, np.ndarray]:
    """
    Calculate SNR using Fourier transform approach.
    
    Signal = low-frequency components (main patterns)
    Noise = high-frequency components (random variations)
    
    Args:
        weights: Weight matrix
        low_freq_cutoff: Fraction of frequencies to consider as signal (0-1)
        
    Returns:
        Tuple of (SNR in dB, frequency spectrum)
    """
    # 2D FFT
    freq_spectrum = np.fft.fft2(weights)
    freq_power = np.abs(freq_spectrum) ** 2
    
    # Separate low and high frequencies
    h, w = freq_power.shape
    cutoff_h = int(h * low_freq_cutoff)
    cutoff_w = int(w * low_freq_cutoff)
    
    # Low frequencies (signal) - center of spectrum
    low_freq_power = freq_power[:cutoff_h, :cutoff_w].sum()
    
    # High frequencies (noise) - edges of spectrum
    high_freq_power = freq_power.sum() - low_freq_power
    
    if high_freq_power == 0:
        return float('inf'), freq_spectrum
    
    # SNR in dB
    snr = 10 * np.log10(low_freq_power / high_freq_power)
    
    return snr, freq_spectrum


def analyze_connection_importance(weights: np.ndarray, 
                                  pre_activity: np.ndarray,
                                  post_activity: np.ndarray) -> np.ndarray:
    """
    Analyze which connections carry important information.
    
    Combines weight magnitude, mutual information, and entropy.
    
    Args:
        weights: Connection weight matrix
        pre_activity: Pre-synaptic neuron activity
        post_activity: Post-synaptic neuron activity
        
    Returns:
        Importance score for each connection (0-1)
    """
    importance = np.zeros_like(weights)
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i, j] != 0:
                # Factor 1: Weight magnitude
                weight_score = np.abs(weights[i, j])
                
                # Factor 2: Mutual information
                if len(pre_activity) > 0 and len(post_activity) > 0:
                    mi = mutual_information(
                        pre_activity[:, i:i+1], 
                        post_activity[:, j:j+1]
                    )
                    mi_score = mi / 10.0  # Normalize roughly
                else:
                    mi_score = 0
                
                # Combined importance
                importance[i, j] = weight_score * (1 + mi_score)
    
    # Normalize to 0-1
    if importance.max() > 0:
        importance = importance / importance.max()
    
    return importance


def prune_by_information(weights: np.ndarray, 
                        keep_fraction: float = 0.1) -> np.ndarray:
    """
    Prune connections keeping only high-information ones.
    
    Uses Shannon entropy to identify signal vs noise.
    
    Args:
        weights: Weight matrix
        keep_fraction: Fraction of connections to keep (0-1)
        
    Returns:
        Pruned weight matrix
    """
    # Calculate importance of each weight
    importance = np.abs(weights)
    
    # Calculate entropy in local neighborhoods
    h, w = weights.shape
    kernel_size = 5
    local_entropy = np.zeros_like(weights)
    
    for i in range(h):
        for j in range(w):
            # Extract local neighborhood
            i_start = max(0, i - kernel_size // 2)
            i_end = min(h, i + kernel_size // 2 + 1)
            j_start = max(0, j - kernel_size // 2)
            j_end = min(w, j + kernel_size // 2 + 1)
            
            neighborhood = weights[i_start:i_end, j_start:j_end]
            
            # Calculate local entropy
            if neighborhood.size > 0:
                local_entropy[i, j] = shannon_entropy(neighborhood)
    
    # Combine magnitude and entropy (keep high magnitude, low entropy)
    score = importance / (local_entropy + 1.0)
    
    # Keep top fraction
    threshold = np.percentile(score[score > 0], (1 - keep_fraction) * 100)
    
    pruned_weights = weights.copy()
    pruned_weights[score < threshold] = 0
    
    return pruned_weights
