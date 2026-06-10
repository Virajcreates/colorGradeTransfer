"""
Color transfer algorithms — four tiers of increasing quality.

Tier 1: Statistical (Reinhard et al.) — fast per-channel mean/std matching
Tier 2: Histogram matching — CDF-based distribution matching
Tier 3: Multi-Scale — histogram matching at multiple resolutions
Tier 4: MKL (Monge-Kantorovitch Linear) — optimal linear color transform
         Captures cross-channel correlations that Reinhard misses.

All functions operate on LAB float32 images.
By default, only chrominance (A, B) channels are transferred.
Luminance (L) is preserved to avoid brightness distortion.
"""

import cv2
import numpy as np
from scipy import linalg
from skimage.exposure import match_histograms
from typing import Optional, Callable
import time


# ---------------------------------------------------------------------------
# Gamut Mapping — prevents out-of-range colors after transfer
# ---------------------------------------------------------------------------

def _soft_clamp(values: np.ndarray, low: float, high: float,
                margin: float = 10.0) -> np.ndarray:
    """
    Soft clamping using sigmoid falloff near boundaries.
    Prevents the harsh color artifacts that hard clipping creates.
    """
    result = values.copy()

    # Soft rolloff near upper bound
    mask_high = values > (high - margin)
    if mask_high.any():
        x = (values[mask_high] - (high - margin)) / margin
        result[mask_high] = (high - margin) + margin * (2.0 / (1.0 + np.exp(-2 * x)) - 1.0)

    # Soft rolloff near lower bound
    mask_low = values < (low + margin)
    if mask_low.any():
        x = ((low + margin) - values[mask_low]) / margin
        result[mask_low] = (low + margin) - margin * (2.0 / (1.0 + np.exp(-2 * x)) - 1.0)

    return result


def _gamut_map_lab(img_lab: np.ndarray) -> np.ndarray:
    """
    Map LAB values back into valid gamut range with soft clamping.
    OpenCV LAB ranges: L [0-255], A [0-255], B [0-255]
    (where 128 is the neutral point for A and B)
    """
    result = img_lab.copy()
    result[:, :, 0] = _soft_clamp(result[:, :, 0], 0, 255, margin=15)
    result[:, :, 1] = _soft_clamp(result[:, :, 1], 0, 255, margin=20)
    result[:, :, 2] = _soft_clamp(result[:, :, 2], 0, 255, margin=20)
    return result


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute the matrix square root using eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    # Clamp to avoid negative eigenvalues from numerical noise
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt_inv(M: np.ndarray) -> np.ndarray:
    """Compute the inverse matrix square root using eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Tier 1 — Reinhard Statistical Transfer
# ---------------------------------------------------------------------------

def reinhard_transfer(source_lab: np.ndarray,
                      reference_lab: np.ndarray,
                      transfer_l: bool = False) -> np.ndarray:
    """
    Reinhard et al. (2001) color transfer.
    Matches mean and standard deviation of LAB channels.
    Only transfers A and B by default (preserving luminance).

    Args:
        source_lab: Source image in LAB float32.
        reference_lab: Reference image in LAB float32.
        transfer_l: If True, also transfer L channel.

    Returns:
        Transferred image in LAB float32.
    """
    result = source_lab.copy()
    start_ch = 0 if transfer_l else 1

    for ch in range(start_ch, 3):
        src_mean = source_lab[:, :, ch].mean()
        src_std = source_lab[:, :, ch].std()
        ref_mean = reference_lab[:, :, ch].mean()
        ref_std = reference_lab[:, :, ch].std()

        if src_std < 1e-6:
            src_std = 1e-6

        # Limit std ratio to prevent extreme amplification
        std_ratio = np.clip(ref_std / src_std, 0.3, 3.0)

        result[:, :, ch] = ((source_lab[:, :, ch] - src_mean)
                            * std_ratio + ref_mean)

    return _gamut_map_lab(result)


# ---------------------------------------------------------------------------
# Tier 2 — Histogram Matching
# ---------------------------------------------------------------------------

def histogram_transfer(source_lab: np.ndarray,
                       reference_lab: np.ndarray,
                       transfer_l: bool = False) -> np.ndarray:
    """
    Full histogram matching on LAB channels using CDF matching.
    By default only matches A and B (chrominance).

    Args:
        source_lab: Source image in LAB float32.
        reference_lab: Reference image in LAB float32.
        transfer_l: If True, match all channels.

    Returns:
        Transferred image in LAB float32.
    """
    if transfer_l:
        matched = match_histograms(source_lab, reference_lab, channel_axis=-1)
        return _gamut_map_lab(matched.astype(np.float32))

    result = source_lab.copy()
    for ch in [1, 2]:
        result[:, :, ch] = match_histograms(
            source_lab[:, :, ch],
            reference_lab[:, :, ch]
        ).astype(np.float32)

    return _gamut_map_lab(result)


# ---------------------------------------------------------------------------
# Tier 3 — Multi-Scale Histogram Transfer
# ---------------------------------------------------------------------------

def multiscale_transfer(source_lab: np.ndarray,
                        reference_lab: np.ndarray,
                        scales: int = 3) -> np.ndarray:
    """
    Multi-scale color transfer: applies histogram matching at multiple
    resolutions and blends the results. Produces smoother, more natural
    color transitions than single-scale matching.

    Args:
        source_lab: Source image in LAB float32.
        reference_lab: Reference image in LAB float32.
        scales: Number of resolution scales to blend.

    Returns:
        Transferred image in LAB float32.
    """
    h, w = source_lab.shape[:2]
    accumulated = np.zeros_like(source_lab)
    weight_sum = 0.0

    for s in range(scales):
        scale_factor = 1.0 / (2 ** s)
        new_h = max(4, int(h * scale_factor))
        new_w = max(4, int(w * scale_factor))

        src_scaled = cv2.resize(source_lab, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)
        ref_scaled = cv2.resize(reference_lab, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)

        transferred = histogram_transfer(src_scaled, ref_scaled,
                                         transfer_l=False)

        upscaled = cv2.resize(transferred, (w, h),
                              interpolation=cv2.INTER_LINEAR)

        weight = 1.0 / (s + 1)
        accumulated += upscaled * weight
        weight_sum += weight

    result = accumulated / weight_sum
    result[:, :, 0] = source_lab[:, :, 0]  # Restore original luminance

    return _gamut_map_lab(result)


# ---------------------------------------------------------------------------
# Tier 4 — MKL (Monge-Kantorovitch Linear) Optimal Color Transfer
# ---------------------------------------------------------------------------

def mkl_transfer(source_lab: np.ndarray,
                 reference_lab: np.ndarray,
                 transfer_l: bool = False) -> np.ndarray:
    """
    Monge-Kantorovitch Linear color transfer.

    This is mathematically the OPTIMAL linear color transform — it finds
    the affine mapping that minimizes the Wasserstein-2 distance between
    the source and reference color distributions.

    Unlike Reinhard (per-channel only), MKL captures cross-channel
    correlations. For example, if the reference has warm highlights
    and cool shadows, MKL will reproduce that coupling.

    The algorithm:
        1. Compute mean and covariance of source and reference colors
        2. Find transform T = Σ_ref^(1/2) @ Σ_src^(-1/2)
        3. Apply: result = (source - μ_src) @ T^T + μ_ref

    Args:
        source_lab: Source image in LAB float32.
        reference_lab: Reference image in LAB float32.
        transfer_l: If True, transform all 3 channels. If False (default),
                    only transform A and B chrominance channels.

    Returns:
        Transferred image in LAB float32.
    """
    h, w = source_lab.shape[:2]

    if transfer_l:
        # Full 3-channel transform
        src_pixels = source_lab.reshape(-1, 3).astype(np.float64)
        ref_pixels = reference_lab.reshape(-1, 3).astype(np.float64)
    else:
        # Chrominance-only (A, B) — 2 channels
        src_pixels = source_lab[:, :, 1:3].reshape(-1, 2).astype(np.float64)
        ref_pixels = reference_lab[:, :, 1:3].reshape(-1, 2).astype(np.float64)

    # Compute means
    src_mean = src_pixels.mean(axis=0)
    ref_mean = ref_pixels.mean(axis=0)

    # Compute covariance matrices
    src_centered = src_pixels - src_mean
    ref_centered = ref_pixels - ref_mean

    # Add small regularization for numerical stability
    n_channels = src_pixels.shape[1]
    eps = 1e-6 * np.eye(n_channels)

    cov_src = (src_centered.T @ src_centered) / len(src_pixels) + eps
    cov_ref = (ref_centered.T @ ref_centered) / len(ref_pixels) + eps

    # Compute the optimal transport matrix
    # T = Σ_ref^(1/2) @ Σ_src^(-1/2)
    src_sqrt_inv = _matrix_sqrt_inv(cov_src)
    ref_sqrt = _matrix_sqrt(cov_ref)
    T = ref_sqrt @ src_sqrt_inv

    # Apply transform
    transformed = (src_centered @ T.T) + ref_mean

    # Reconstruct image
    result = source_lab.copy()
    if transfer_l:
        result = transformed.reshape(h, w, 3).astype(np.float32)
    else:
        result[:, :, 1:3] = transformed.reshape(h, w, 2).astype(np.float32)

    return _gamut_map_lab(result)


# ---------------------------------------------------------------------------
# Tier 3 — Neural Style Transfer
# ---------------------------------------------------------------------------

def neural_transfer(source_rgb: np.ndarray,
                    reference_rgb: np.ndarray,
                    iterations: int = 100,
                    style_weight: float = 1e5,
                    content_weight: float = 1.0,
                    device: str = 'cpu') -> np.ndarray:
    """
    Neural Style Transfer (Gatys et al.) modified for color grading.
    Optimizes the input image pixels to match style/color of reference image.

    Args:
        source_rgb: Source image in RGB uint8 (H, W, 3).
        reference_rgb: Reference image in RGB uint8 (H, W, 3).
        iterations: Number of optimization iterations.
        style_weight: Weight of style loss.
        content_weight: Weight of content loss.
        device: Target hardware ('cuda' or 'cpu').

    Returns:
        Transferred image in RGB uint8 (H, W, 3).
    """
    import torch
    import torch.nn.functional as F
    from core.extractor import VGGFeatureExtractor

    dev = torch.device(device)

    # Initialize feature extractor
    extractor = VGGFeatureExtractor(device=dev)
    extractor.features.float()

    # Preprocess and extract target features
    content_features = {k: v.float() for k, v in extractor.extract_features(source_rgb).items()}
    style_grams = {k: v.float() for k, v in extractor.extract_style_features(reference_rgb).items()}

    content_layer = 19  # conv4_1 in VGG19
    style_layers = list(style_grams.keys())

    # Initialize output image as copy of source
    output_tensor = extractor.preprocess(source_rgb).float().clone().requires_grad_(True)

    # Use Adam optimizer
    optimizer = torch.optim.Adam([output_tensor], lr=0.02)

    # Optimization loop
    for _ in range(iterations):
        optimizer.zero_grad()

        # Forward pass through VGG feature extractor
        x = output_tensor
        current_features = {}
        for idx, layer in enumerate(extractor.features):
            x = layer(x)
            if idx in extractor.layer_indices:
                current_features[idx] = x

        # Compute content loss (distance from source content features)
        content_loss = F.mse_loss(current_features[content_layer], content_features[content_layer])

        # Compute style loss
        style_loss = torch.tensor(0.0, device=dev)
        for layer_idx in style_layers:
            current_gram = extractor.gram_matrix(current_features[layer_idx])
            target_gram = style_grams[layer_idx]
            style_loss += F.mse_loss(current_gram, target_gram)
        style_loss /= len(style_layers)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        # Clip gradient norm to stabilize
        torch.nn.utils.clip_grad_norm_([output_tensor], max_norm=1.0)
        optimizer.step()

        # Project back to valid range
        with torch.no_grad():
            output_tensor.clamp_(-2.5, 2.5)

    # Postprocess: convert output tensor back to numpy RGB uint8
    with torch.no_grad():
        out_img = output_tensor.squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        out_img = out_img * std + mean
        out_img = torch.clamp(out_img, 0.0, 1.0)
        out_img = out_img.permute(1, 2, 0).numpy()
        result_rgb = (out_img * 255.0).astype(np.uint8)

    return result_rgb

