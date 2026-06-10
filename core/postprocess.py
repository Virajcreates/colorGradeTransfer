"""
Postprocessing pipeline for color grade transfer output.
Handles luminance preservation, chrominance smoothing, saturation control,
sharpening, denoising, and strength blending.
"""

import cv2
import numpy as np
from typing import Optional


def preserve_luminance(source_lab: np.ndarray,
                       transferred_lab: np.ndarray,
                       blend: float = 1.0) -> np.ndarray:
    """
    Luminance-preserving blend: keep the L channel from source,
    only use transferred A and B channels.

    Args:
        source_lab: Original source in LAB float32.
        transferred_lab: Transferred image in LAB float32.
        blend: 1.0 = full source luminance, 0.0 = full transferred luminance.

    Returns:
        LAB float32 image with source luminance.
    """
    result = transferred_lab.copy()
    result[:, :, 0] = (source_lab[:, :, 0] * blend +
                       transferred_lab[:, :, 0] * (1.0 - blend))
    return result


def control_saturation(img_lab: np.ndarray,
                       source_lab: np.ndarray,
                       max_boost: float = 1.5) -> np.ndarray:
    """
    Prevent excessive saturation boost. If the transfer amplified
    the chrominance channels too much, pull them back toward the
    source's saturation level.

    Args:
        img_lab: Transferred image in LAB float32.
        source_lab: Original source in LAB float32.
        max_boost: Maximum allowed saturation multiplier (1.5 = 50% boost max).

    Returns:
        Saturation-controlled LAB float32 image.
    """
    result = img_lab.copy()

    # Compute saturation as distance from neutral (128) in A-B plane
    src_sat = np.sqrt((source_lab[:, :, 1] - 128.0)**2 +
                      (source_lab[:, :, 2] - 128.0)**2)
    res_sat = np.sqrt((result[:, :, 1] - 128.0)**2 +
                      (result[:, :, 2] - 128.0)**2)

    # Where saturation was boosted beyond max_boost, scale it back
    safe_src = np.maximum(src_sat, 1.0)  # avoid div by zero
    boost = res_sat / safe_src

    # Create scale map: 1.0 where fine, reduced where over-boosted
    scale = np.where(boost > max_boost, max_boost / boost, 1.0)

    # Apply scale to A and B channels (relative to neutral 128)
    result[:, :, 1] = 128.0 + (result[:, :, 1] - 128.0) * scale
    result[:, :, 2] = 128.0 + (result[:, :, 2] - 128.0) * scale

    return result


def unsharp_mask(img_bgr: np.ndarray,
                 sigma: float = 1.0,
                 strength: float = 0.3,
                 threshold: int = 0) -> np.ndarray:
    """
    Apply unsharp mask to recover edge crispness after color transfer.

    Args:
        img_bgr: BGR uint8 image.
        sigma: Gaussian blur sigma.
        strength: Sharpening strength (0.0 = none, 1.0 = strong).
        threshold: Minimum difference to apply sharpening.

    Returns:
        Sharpened BGR uint8 image.
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)

    if threshold > 0:
        diff = cv2.absdiff(img_bgr, blurred)
        mask = (diff > threshold).astype(np.float32)
        sharpened = (img_bgr * (1 - mask) + sharpened * mask).astype(np.uint8)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img_bgr: np.ndarray,
            h: float = 5.0,
            h_color: float = 5.0,
            template_window: int = 7,
            search_window: int = 21) -> np.ndarray:
    """
    Non-Local Means denoising for color noise reduction.

    Args:
        img_bgr: BGR uint8 image.
        h: Filter strength for luminance.
        h_color: Filter strength for color.
        template_window: Template patch size (odd).
        search_window: Search area size (odd).

    Returns:
        Denoised BGR uint8 image.
    """
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h, h_color, template_window, search_window
    )


def blend_strength(source_bgr: np.ndarray,
                   transferred_bgr: np.ndarray,
                   strength: float = 0.8) -> np.ndarray:
    """
    Blend between original and transferred image.

    Args:
        source_bgr: Original BGR uint8.
        transferred_bgr: Transferred BGR uint8.
        strength: 0.0 = original, 1.0 = fully transferred.

    Returns:
        Blended BGR uint8 image.
    """
    strength = np.clip(strength, 0.0, 1.0)
    blended = cv2.addWeighted(
        transferred_bgr, strength,
        source_bgr, 1.0 - strength,
        0
    )
    return blended.astype(np.uint8)


def smooth_chrominance(transferred_lab: np.ndarray,
                       source_lab: np.ndarray,
                       sigma_color: float = 50.0,
                       sigma_space: float = 12.0,
                       iterations: int = 2) -> np.ndarray:
    """
    Edge-aware chrominance smoothing to remove color splotch artifacts.

    Histogram matching can create spatially noisy color (e.g., random
    pink/green splotches) because it maps each pixel independently.
    This function smooths the A and B channels using a bilateral filter
    guided by the source luminance — preserving color edges that align
    with structural edges, while eliminating false color patches.

    Applied multiple times for thorough artifact removal.

    Args:
        transferred_lab: Transferred image in LAB float32.
        source_lab: Original source in LAB float32.
        sigma_color: Bilateral filter color sigma (higher = more smoothing).
        sigma_space: Bilateral filter spatial sigma (higher = wider area).
        iterations: Number of smoothing passes.

    Returns:
        Smoothed LAB float32 image.
    """
    result = transferred_lab.copy()

    d = int(sigma_space * 2) | 1  # diameter must be odd

    for _ in range(iterations):
        # Convert to uint8 for bilateral filter
        a_ch = np.clip(result[:, :, 1], 0, 255).astype(np.uint8)
        b_ch = np.clip(result[:, :, 2], 0, 255).astype(np.uint8)

        # Bilateral filter smooths color while preserving edges
        a_smooth = cv2.bilateralFilter(a_ch, d, sigma_color, sigma_space)
        b_smooth = cv2.bilateralFilter(b_ch, d, sigma_color, sigma_space)

        result[:, :, 1] = a_smooth.astype(np.float32)
        result[:, :, 2] = b_smooth.astype(np.float32)

    return result


def postprocess(source_lab: np.ndarray,
                transferred_lab: np.ndarray,
                strength: float = 0.8,
                preserve_lum: bool = True,
                saturation_cap: float = 1.5,
                sharpen: bool = True,
                sharpen_sigma: float = 1.0,
                sharpen_strength: float = 0.3,
                apply_denoise: bool = False,
                denoise_strength: float = 5.0) -> np.ndarray:
    """
    Full postprocessing pipeline.

    Steps:
        1. Luminance preservation (keeps source brightness)
        2. Saturation control (prevents oversaturation)
        3. Chrominance smoothing (removes color splotch artifacts)
        4. Convert LAB -> BGR
        5. Blend with original at given strength
        6. Unsharp mask (optional)
        7. Denoising (optional)

    Args:
        source_lab: Original source in LAB float32.
        transferred_lab: Transferred result in LAB float32.
        strength: Blend strength (0-1).
        preserve_lum: Whether to preserve source luminance.
        saturation_cap: Maximum saturation boost allowed (1.5 = 50%).
        sharpen: Whether to apply unsharp mask.
        sharpen_sigma: Sharpening Gaussian sigma.
        sharpen_strength: Sharpening intensity.
        apply_denoise: Whether to apply NL-means denoising.
        denoise_strength: Denoising filter strength.

    Returns:
        Final output as BGR uint8 numpy array.
    """
    from core.preprocess import lab_to_bgr

    # Step 1: Luminance preservation
    if preserve_lum:
        transferred_lab = preserve_luminance(source_lab, transferred_lab)

    # Step 2: Saturation control
    transferred_lab = control_saturation(transferred_lab, source_lab,
                                         max_boost=saturation_cap)

    # Step 3: Chrominance smoothing — removes color splotch artifacts
    transferred_lab = smooth_chrominance(transferred_lab, source_lab)

    # Step 4: Convert to BGR
    source_bgr = lab_to_bgr(source_lab)
    transferred_bgr = lab_to_bgr(transferred_lab)

    # Step 5: Blend
    result = blend_strength(source_bgr, transferred_bgr, strength)

    # Step 6: Sharpen
    if sharpen:
        result = unsharp_mask(result, sigma=sharpen_sigma,
                              strength=sharpen_strength)

    # Step 7: Denoise
    if apply_denoise:
        result = denoise(result, h=denoise_strength,
                         h_color=denoise_strength)

    return result
