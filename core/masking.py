"""
Mask-guided color transfer.

Allows applying color transfer to specific regions of the image
using either:
  - Auto-detected masks (sky, foreground, shadows, highlights)
  - User-uploaded custom masks
  - Luminance-based range selection

The mask defines WHERE the transfer is applied (white = transfer,
black = keep original), with smooth blending at edges.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def create_sky_mask(image_rgb: np.ndarray,
                    threshold: float = 0.4) -> np.ndarray:
    """
    Auto-detect sky region using flood-fill from top edge +
    texture/position heuristics. Works for ALL sky colors
    (blue, orange sunset, pink, overcast gray, etc.).

    Strategy:
        1. Score each pixel by position (upper = more likely sky),
           texture smoothness, and brightness.
        2. Use the top rows as seed region and grow downward
           using color similarity.
        3. Combine both approaches for a robust mask.

    Args:
        image_rgb: Input image as RGB uint8.
        threshold: Sensitivity (0-1, lower = more sky detected).

    Returns:
        Soft mask as float32 [0, 1] (1 = sky region).
    """
    h, w = image_rgb.shape[:2]

    # --- Score 1: Position bias (upper portion) ---
    y_weight = np.linspace(1.0, 0.0, h).reshape(-1, 1)
    y_weight = np.power(y_weight, 0.7)  # Softer falloff
    position_score = np.broadcast_to(y_weight, (h, w)).copy()

    # --- Score 2: Low texture (sky = smooth gradient) ---
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Use Laplacian variance as texture measure
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # Local texture energy via block averaging
    texture_energy = cv2.blur(np.abs(lap), (15, 15))
    # Normalize and invert
    tex_max = np.percentile(texture_energy, 95)
    if tex_max > 0:
        texture_score = 1.0 - np.clip(texture_energy / tex_max, 0, 1)
    else:
        texture_score = np.ones((h, w), dtype=np.float32)

    # --- Score 3: Brightness ---
    val = gray.astype(np.float32) / 255.0
    bright_score = val

    # --- Score 4: Color similarity to top strip ---
    # Sample the top 5% of the image as reference sky color
    top_strip = image_rgb[:max(h // 20, 5), :, :]
    sky_mean = top_strip.reshape(-1, 3).mean(axis=0).astype(np.float32)
    sky_std = max(top_strip.reshape(-1, 3).std(), 15.0)

    # Color distance from sky reference
    img_f = image_rgb.astype(np.float32)
    color_dist = np.sqrt(np.sum((img_f - sky_mean) ** 2, axis=2))
    # Normalize: close to sky color = high score
    color_score = np.exp(-(color_dist ** 2) / (2 * (sky_std * 3) ** 2))

    # --- Combine all scores ---
    combined = (position_score * 0.25 +
                texture_score * 0.30 +
                bright_score * 0.15 +
                color_score * 0.30)

    # --- Flood-fill refinement from top edge ---
    # Start from pixels in the top 10% that score well
    seed_rows = max(h // 10, 10)
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    seed_region = combined[:seed_rows, :]
    seed_mask[:seed_rows, :] = (seed_region > 0.45).astype(np.uint8) * 255

    # Grow the mask downward using connected components
    # Use the combined score as a probability map
    prob_mask = (combined > threshold).astype(np.uint8) * 255

    # Connect seed to the probability mask via morphological propagation
    # Start with seed and iteratively grow into high-probability areas
    grown = seed_mask.copy()
    kernel_grow = np.ones((5, 5), np.uint8)
    for _ in range(h // 10):
        dilated = cv2.dilate(grown, kernel_grow, iterations=1)
        grown = cv2.bitwise_and(dilated, prob_mask)
        # Keep connected to seed
        grown = cv2.bitwise_or(grown, seed_mask)

    # Also include the raw high-confidence pixels
    high_conf = (combined > 0.65).astype(np.uint8) * 255
    # But only if they're in the upper 60% and connected
    upper_only = np.zeros_like(high_conf)
    upper_only[:int(h * 0.6), :] = high_conf[:int(h * 0.6), :]
    grown = cv2.bitwise_or(grown, upper_only)

    # --- Morphological cleanup ---
    kernel_close = np.ones((21, 21), np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((9, 9), np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, kernel_open)

    # --- Smooth edges for natural blending ---
    mask_smooth = cv2.GaussianBlur(grown.astype(np.float32), (41, 41), 15)
    mask_max = mask_smooth.max()
    if mask_max > 0:
        mask_smooth = mask_smooth / mask_max
    else:
        mask_smooth = np.zeros((h, w), dtype=np.float32)

    return mask_smooth


def create_foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Auto-detect foreground using GrabCut with center prior.

    Args:
        image_rgb: Input image as RGB uint8.

    Returns:
        Binary mask as float32 [0, 1] (1 = foreground).
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Initialize with a rectangle covering the center 60% of the image
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.15)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,
                5, cv2.GC_INIT_WITH_RECT)

    # Convert to binary (foreground + probable foreground = 1)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       1.0, 0.0).astype(np.float32)

    # Smooth edges
    fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 7)
    fg_mask = fg_mask / max(fg_mask.max(), 1e-6)

    return fg_mask


def create_luminance_mask(image_rgb: np.ndarray,
                          range_type: str = "highlights") -> np.ndarray:
    """
    Create a mask based on luminance ranges.

    Args:
        image_rgb: Input image as RGB uint8.
        range_type: One of "highlights", "midtones", "shadows".

    Returns:
        Soft mask as float32 [0, 1].
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if range_type == "highlights":
        # Smooth ramp: 0 below 0.6, 1 above 0.85
        mask = np.clip((gray - 0.6) / 0.25, 0, 1)
    elif range_type == "shadows":
        # Smooth ramp: 1 below 0.15, 0 above 0.4
        mask = np.clip((0.4 - gray) / 0.25, 0, 1)
    elif range_type == "midtones":
        # Bell curve centered at 0.5
        mask = np.exp(-((gray - 0.5) ** 2) / (2 * 0.15 ** 2))
    else:
        raise ValueError(f"Unknown range_type: {range_type}. "
                         f"Use 'highlights', 'midtones', or 'shadows'.")

    # Smooth for natural blending
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    return mask.astype(np.float32)


def load_custom_mask(mask_input: np.ndarray,
                     target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load and normalize a user-provided mask image.

    Args:
        mask_input: Mask image (any format — RGB, grayscale, etc.).
        target_size: (width, height) to resize to.

    Returns:
        Normalized mask as float32 [0, 1].
    """
    if len(mask_input.shape) == 3:
        mask = cv2.cvtColor(mask_input, cv2.COLOR_RGB2GRAY)
    else:
        mask = mask_input.copy()

    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
    mask = mask.astype(np.float32) / 255.0

    return mask


def apply_masked_transfer(source_rgb: np.ndarray,
                          transferred_rgb: np.ndarray,
                          mask: np.ndarray,
                          feather: float = 10.0) -> np.ndarray:
    """
    Blend source and transferred images using a mask.

    White (1.0) = show transferred, Black (0.0) = show original.

    Args:
        source_rgb: Original image as RGB uint8.
        transferred_rgb: Transferred image as RGB uint8.
        mask: Float32 mask [0, 1] with same H, W as images.
        feather: Additional Gaussian blur radius for edge softness.

    Returns:
        Blended image as RGB uint8.
    """
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Resize mask to match image dimensions
    h, w = source_rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Additional feathering
    if feather > 0:
        ksize = int(feather * 2) | 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    # Expand mask to 3 channels
    mask_3ch = mask[:, :, np.newaxis]

    # Blend
    result = (transferred_rgb.astype(np.float32) * mask_3ch +
              source_rgb.astype(np.float32) * (1.0 - mask_3ch))

    return np.clip(result, 0, 255).astype(np.uint8)


def create_mask_preview(image_rgb: np.ndarray,
                        mask: np.ndarray,
                        color: Tuple[int, int, int] = (102, 126, 234)
                        ) -> np.ndarray:
    """
    Create a preview showing the mask overlay on the image.

    Args:
        image_rgb: Original image as RGB uint8.
        mask: Float32 mask [0, 1].
        color: Overlay color (R, G, B).

    Returns:
        Preview image as RGB uint8 with colored mask overlay.
    """
    h, w = image_rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = np.zeros_like(image_rgb, dtype=np.float32)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]

    mask_3ch = mask[:, :, np.newaxis]

    preview = (image_rgb.astype(np.float32) * (1 - mask_3ch * 0.4) +
               overlay * mask_3ch * 0.4)

    return np.clip(preview, 0, 255).astype(np.uint8)
