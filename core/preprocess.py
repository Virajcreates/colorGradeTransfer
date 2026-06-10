"""
Preprocessing pipeline for color grade transfer.
Handles image loading, resizing, color space conversion (BGR <-> LAB),
and input normalization from multiple sources (file, numpy, PIL, Gradio).

LAB color space is the foundation of every color transfer approach because
it cleanly separates luminance (L) from chrominance (A, B channels).
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------

def load_image(image_input: Union[str, np.ndarray, Image.Image],
               size: Optional[Tuple[int, int]] = (512, 512),
               keep_aspect: bool = True,
               pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Load an image from a file path, numpy array, or PIL Image.
    Handles RGBA, grayscale, and various input formats gracefully.

    Args:
        image_input: File path string, numpy array (RGB uint8), or PIL Image.
        size: Target (width, height) to resize to. None = keep original size.
        keep_aspect: If True, resize preserving aspect ratio and pad.
                     If False, stretch to exact size.
        pad_color: BGR color to use for padding when keep_aspect=True.

    Returns:
        BGR uint8 numpy array.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_input}")
        img = _normalize_channels(img)

    elif isinstance(image_input, Image.Image):
        # Convert PIL to RGB first, then to BGR
        pil_rgb = image_input.convert("RGB")
        img = np.array(pil_rgb)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif isinstance(image_input, np.ndarray):
        img = _normalize_channels(image_input.copy())
        # Numpy arrays from Gradio/PIL are RGB — convert to BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Resize
    if size is not None:
        if keep_aspect:
            img = _resize_with_aspect(img, size, pad_color)
        else:
            img = cv2.resize(img, size, interpolation=_pick_interpolation(img, size))

    return img


def _normalize_channels(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 3-channel BGR.
    Handles grayscale (1 or 2D), RGBA (4-channel), and already-3-channel images.
    """
    if len(img.shape) == 2:
        # Grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # RGBA -> BGR (drop alpha, blend onto white)
            img = _rgba_to_bgr(img)
        elif img.shape[2] == 1:
            # Single channel -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _rgba_to_bgr(img_rgba: np.ndarray,
                 bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Composite RGBA onto a solid background color.
    Default background is white — avoids black halos around transparent regions.
    """
    alpha = img_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = img_rgba[:, :, :3].astype(np.float32)
    bg = np.full_like(rgb, bg_color, dtype=np.float32)
    composited = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return composited


def _resize_with_aspect(img: np.ndarray,
                        target_size: Tuple[int, int],
                        pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize preserving aspect ratio, then pad to exact target size.
    Image is centered within the padded canvas.
    """
    tw, th = target_size
    h, w = img.shape[:2]

    # Compute scale to fit within target
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = _pick_interpolation(img, (new_w, new_h))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Create padded canvas and center the image
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def _pick_interpolation(img: np.ndarray,
                        target_size: Tuple[int, int]) -> int:
    """
    Pick the best interpolation method based on whether we're
    upscaling or downscaling.
    """
    h, w = img.shape[:2]
    tw, th = target_size
    if tw * th < w * h:
        return cv2.INTER_AREA       # Downscaling — INTER_AREA avoids aliasing
    else:
        return cv2.INTER_LANCZOS4   # Upscaling — Lanczos for sharp results


# ---------------------------------------------------------------------------
# Color Space Conversions
# ---------------------------------------------------------------------------

def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to LAB float32.

    LAB color space separates luminance (L) from chrominance (A, B),
    making it ideal for color transfer operations.

    OpenCV LAB ranges (uint8 encoding):
        L: 0–255  (maps to 0–100 in true LAB)
        A: 0–255  (maps to -128 to +127)
        B: 0–255  (maps to -128 to +127)

    Args:
        img_bgr: BGR uint8 numpy array.

    Returns:
        LAB float32 numpy array with OpenCV uint8-scaled ranges.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return img_lab


def lab_to_bgr(img_lab: np.ndarray) -> np.ndarray:
    """
    Convert a LAB float32 image back to BGR uint8.

    Args:
        img_lab: LAB float32 numpy array (OpenCV uint8-scaled ranges).

    Returns:
        BGR uint8 numpy array.
    """
    img_lab_clipped = np.clip(img_lab, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_lab_clipped, cv2.COLOR_LAB2BGR)
    return img_bgr


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB (for display / Gradio output)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR (from Gradio input)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# LAB Channel Utilities
# ---------------------------------------------------------------------------

def split_lab(img_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a LAB image into individual L, A, B channels.

    Returns:
        Tuple of (L, A, B) — each a 2D float32 array.
    """
    return img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]


def merge_lab(l_ch: np.ndarray, a_ch: np.ndarray, b_ch: np.ndarray) -> np.ndarray:
    """
    Merge individual L, A, B channels back into a 3-channel LAB image.

    Args:
        l_ch, a_ch, b_ch: 2D float32 arrays for each channel.

    Returns:
        LAB float32 numpy array (H, W, 3).
    """
    return np.stack([l_ch, a_ch, b_ch], axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def load_and_convert(image_input: Union[str, np.ndarray, Image.Image],
                     size: Optional[Tuple[int, int]] = (512, 512),
                     keep_aspect: bool = False) -> np.ndarray:
    """
    Convenience function: load an image and convert directly to LAB float32.

    Args:
        image_input: File path, numpy array (RGB), or PIL Image.
        size: Target (width, height). None = keep original.
        keep_aspect: Preserve aspect ratio with padding.

    Returns:
        LAB float32 numpy array.
    """
    img_bgr = load_image(image_input, size, keep_aspect=keep_aspect)
    img_lab = bgr_to_lab(img_bgr)
    return img_lab


def preprocess_pair(source_input: Union[str, np.ndarray, Image.Image],
                    reference_input: Union[str, np.ndarray, Image.Image],
                    size: Tuple[int, int] = (512, 512),
                    keep_aspect: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a source-reference image pair for color transfer.
    Returns both LAB conversions AND the source BGR (needed for postprocessing).

    Args:
        source_input: Source image (any supported input type).
        reference_input: Reference image (any supported input type).
        size: Target size for both images.
        keep_aspect: Preserve aspect ratio.

    Returns:
        Tuple of (source_lab, reference_lab, source_bgr).
    """
    source_bgr = load_image(source_input, size, keep_aspect=keep_aspect)
    reference_bgr = load_image(reference_input, size, keep_aspect=keep_aspect)

    source_lab = bgr_to_lab(source_bgr)
    reference_lab = bgr_to_lab(reference_bgr)

    return source_lab, reference_lab, source_bgr
