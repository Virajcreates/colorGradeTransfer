"""
Color utility functions — LAB statistics, histogram helpers, and
color space analysis tools used across the pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def channel_stats(img_lab: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute per-channel statistics for a LAB image.

    Args:
        img_lab: LAB float32 numpy array.

    Returns:
        Dict with keys 'L', 'A', 'B', each containing
        'mean', 'std', 'min', 'max'.
    """
    channel_names = ['L', 'A', 'B']
    stats = {}
    for i, name in enumerate(channel_names):
        ch = img_lab[:, :, i]
        stats[name] = {
            'mean': float(ch.mean()),
            'std': float(ch.std()),
            'min': float(ch.min()),
            'max': float(ch.max()),
        }
    return stats


def compute_histogram(img_lab: np.ndarray,
                      bins: int = 256) -> Dict[str, np.ndarray]:
    """
    Compute histograms for each LAB channel.

    Args:
        img_lab: LAB float32 numpy array.
        bins: Number of histogram bins.

    Returns:
        Dict with keys 'L', 'A', 'B', each containing a 1D histogram array.
    """
    channel_names = ['L', 'A', 'B']
    histograms = {}
    for i, name in enumerate(channel_names):
        ch = img_lab[:, :, i].astype(np.uint8)
        hist = cv2.calcHist([ch], [0], None, [bins], [0, 256])
        histograms[name] = hist.flatten()
    return histograms


def color_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Compute the mean CIEDE2000-simplified color distance between two LAB images.
    Uses simple Euclidean distance in LAB space as an approximation.

    Args:
        lab1: First LAB float32 image.
        lab2: Second LAB float32 image.

    Returns:
        Mean Euclidean distance in LAB space.
    """
    diff = lab1.astype(np.float64) - lab2.astype(np.float64)
    pixel_distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(pixel_distances.mean())


def dominant_colors(img_bgr: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Extract dominant colors using k-means clustering.

    Args:
        img_bgr: BGR uint8 image.
        k: Number of clusters.

    Returns:
        Array of shape (k, 3) with dominant BGR colors, sorted by frequency.
    """
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Sort by cluster size (most frequent first)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    centers = centers[sorted_indices]

    return centers.astype(np.uint8)


def auto_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple gray-world auto white balance.
    Adjusts each channel so its mean equals the overall mean.

    Args:
        img_bgr: BGR uint8 image.

    Returns:
        White-balanced BGR uint8 image.
    """
    result = img_bgr.astype(np.float32)
    avg = result.mean()
    for ch in range(3):
        ch_avg = result[:, :, ch].mean()
        if ch_avg > 0:
            result[:, :, ch] *= avg / ch_avg
    return np.clip(result, 0, 255).astype(np.uint8)
