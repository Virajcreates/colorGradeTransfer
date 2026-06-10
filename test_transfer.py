"""
Phase 3 — Color Transfer Engine validation.
Tests all three tiers: Statistical (Reinhard), Histogram matching, and Neural.
Generates visual comparison outputs for each method.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time

from core.preprocess import (
    load_image, load_and_convert, preprocess_pair,
    bgr_to_lab, lab_to_bgr, bgr_to_rgb, rgb_to_bgr,
)
from core.transfer import reinhard_transfer, histogram_transfer, neural_transfer
from core.postprocess import postprocess
from utils.color_utils import channel_stats


def create_realistic_test_images():
    """Generate more realistic test images with gradients and patterns."""
    os.makedirs("test_output", exist_ok=True)

    # Source: Warm landscape (golden hour photo simulation)
    h, w = 512, 512
    source = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient (warm orange-to-pink)
    for y in range(h // 2):
        t = y / (h // 2)
        r = int(255 - 50 * t)
        g = int(180 - 80 * t)
        b = int(100 + 60 * t)
        source[y, :] = [b, g, r]  # BGR

    # Ground (dark warm earth tones)
    for y in range(h // 2, h):
        t = (y - h // 2) / (h // 2)
        r = int(80 - 30 * t)
        g = int(60 - 20 * t)
        b = int(30 - 10 * t)
        source[y, :] = [b, g, r]

    # Add sun
    cv2.circle(source, (380, 120), 45, (80, 200, 255), -1)
    cv2.circle(source, (380, 120), 55, (60, 160, 230), 3)

    # Add tree silhouette
    pts = np.array([[200, 256], [170, 180], [230, 180]], np.int32)
    cv2.fillPoly(source, [pts], (15, 25, 20))
    cv2.rectangle(source, (195, 256), (205, 310), (15, 25, 20), -1)

    cv2.imwrite("test_output/source_warm.png", source)

    # Reference: Cool cyberpunk / teal-orange grade
    reference = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            t_y = y / h
            t_x = x / w
            r = int(30 + 100 * t_x)
            g = int(180 - 80 * t_y)
            b = int(200 + 55 * t_y)
            reference[y, x] = [b, g, r]

    # Add neon grid lines
    for i in range(0, w, 50):
        cv2.line(reference, (i, 0), (i, h), (255, 100, 50), 1)
    for i in range(0, h, 50):
        cv2.line(reference, (0, i), (w, i), (255, 100, 50), 1)

    cv2.imwrite("test_output/reference_cool.png", reference)

    print("  Created source (warm) and reference (cool) test images")
    return source, reference


def test_reinhard(source_lab, reference_lab, source_bgr):
    """Test Tier 1 — Reinhard statistical transfer."""
    print("\n" + "=" * 50)
    print("  TIER 1: Reinhard Statistical Transfer")
    print("=" * 50)

    t0 = time.time()
    transferred_lab = reinhard_transfer(source_lab, reference_lab, transfer_l=True)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed*1000:.1f}ms")

    # Validate output
    assert transferred_lab.shape == source_lab.shape
    assert transferred_lab.dtype == np.float32

    # Check that color statistics shifted toward reference
    src_stats = channel_stats(source_lab)
    ref_stats = channel_stats(reference_lab)
    out_stats = channel_stats(transferred_lab)

    print(f"\n  Channel statistics comparison:")
    print(f"  {'Channel':<10} {'Source':<15} {'Reference':<15} {'Result':<15}")
    print(f"  {'-'*55}")
    for ch in ['L', 'A', 'B']:
        s = f"{src_stats[ch]['mean']:.1f} +/- {src_stats[ch]['std']:.1f}"
        r = f"{ref_stats[ch]['mean']:.1f} +/- {ref_stats[ch]['std']:.1f}"
        o = f"{out_stats[ch]['mean']:.1f} +/- {out_stats[ch]['std']:.1f}"
        print(f"  {ch:<10} {s:<15} {r:<15} {o:<15}")

    # Result mean should be closer to reference than to source (for A and B)
    for ch in ['A', 'B']:
        dist_to_ref = abs(out_stats[ch]['mean'] - ref_stats[ch]['mean'])
        dist_to_src = abs(out_stats[ch]['mean'] - src_stats[ch]['mean'])
        assert dist_to_ref < dist_to_src + 1, \
            f"Channel {ch}: result should be closer to reference"

    print("  PASS - Reinhard transfer shifts colors toward reference")

    # Postprocess and save
    result_bgr = postprocess(source_lab, transferred_lab, strength=1.0,
                             preserve_lum=False, sharpen=False)
    cv2.imwrite("test_output/result_reinhard.png", result_bgr)

    # Also save luminance-preserved version
    result_bgr_lum = postprocess(source_lab, transferred_lab, strength=1.0,
                                 preserve_lum=True, sharpen=True)
    cv2.imwrite("test_output/result_reinhard_lum_preserved.png", result_bgr_lum)

    return transferred_lab


def test_histogram(source_lab, reference_lab, source_bgr):
    """Test Tier 2 — Histogram matching."""
    print("\n" + "=" * 50)
    print("  TIER 2: Histogram Matching")
    print("=" * 50)

    t0 = time.time()
    transferred_lab = histogram_transfer(source_lab, reference_lab, transfer_l=True)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed*1000:.1f}ms")

    # Validate
    assert transferred_lab.shape == source_lab.shape
    assert transferred_lab.dtype == np.float32

    out_stats = channel_stats(transferred_lab)
    ref_stats = channel_stats(reference_lab)

    print(f"\n  Result vs Reference comparison:")
    for ch in ['A', 'B']:
        diff = abs(out_stats[ch]['mean'] - ref_stats[ch]['mean'])
        print(f"  {ch} mean diff from reference: {diff:.1f}")

    print("  PASS - Histogram matching applied successfully")

    # Save
    result_bgr = postprocess(source_lab, transferred_lab, strength=1.0,
                             preserve_lum=False, sharpen=False)
    cv2.imwrite("test_output/result_histogram.png", result_bgr)

    result_bgr_lum = postprocess(source_lab, transferred_lab, strength=1.0,
                                 preserve_lum=True, sharpen=True)
    cv2.imwrite("test_output/result_histogram_lum_preserved.png", result_bgr_lum)

    return transferred_lab


def test_neural(source_bgr, reference_bgr):
    """Test Tier 3 — Neural style transfer (reduced iterations for testing)."""
    print("\n" + "=" * 50)
    print("  TIER 3: Neural Style Transfer")
    print("=" * 50)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    if device == 'cuda':
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"  VRAM before: {mem_before:.1f} MB")

    # Use fewer iterations for testing (50 instead of 250)
    source_rgb = bgr_to_rgb(source_bgr)
    reference_rgb = bgr_to_rgb(reference_bgr)

    t0 = time.time()
    result_rgb = neural_transfer(
        source_rgb, reference_rgb,
        iterations=75,         # Reduced for testing speed
        style_weight=1e5,
        content_weight=1.0,
        device=device
    )
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")

    if device == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"  VRAM after cleanup: {mem_after:.1f} MB")

    # Validate output
    assert result_rgb.shape == source_rgb.shape
    assert result_rgb.dtype == np.uint8

    # Convert to BGR and save
    result_bgr = rgb_to_bgr(result_rgb)
    cv2.imwrite("test_output/result_neural.png", result_bgr)

    print("  PASS - Neural transfer completed successfully")
    return result_rgb


def create_comparison_grid():
    """Create a side-by-side comparison of all methods."""
    print("\n" + "=" * 50)
    print("  Creating comparison grid")
    print("=" * 50)

    labels = ["Source", "Reference", "Reinhard", "Histogram", "Neural"]
    files = [
        "test_output/source_warm.png",
        "test_output/reference_cool.png",
        "test_output/result_reinhard.png",
        "test_output/result_histogram.png",
        "test_output/result_neural.png",
    ]

    images = []
    for f in files:
        if os.path.exists(f):
            img = cv2.imread(f)
            img = cv2.resize(img, (300, 300))
            images.append(img)
        else:
            # Placeholder
            placeholder = np.full((300, 300, 3), 40, dtype=np.uint8)
            cv2.putText(placeholder, "N/A", (120, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
            images.append(placeholder)

    # Add labels
    labeled = []
    for img, label in zip(images, labels):
        # Add label bar on top
        bar = np.full((40, 300, 3), 30, dtype=np.uint8)
        cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        labeled.append(np.vstack([bar, img]))

    grid = np.hstack(labeled)
    cv2.imwrite("test_output/comparison_grid.png", grid)
    print(f"  Saved comparison grid ({grid.shape[1]}x{grid.shape[0]}) to test_output/comparison_grid.png")


def main():
    print("=" * 60)
    print("  Phase 3 - Color Transfer Engine Validation")
    print("=" * 60)

    # Generate test images
    print("\n[0] Generating test images...")
    source_bgr_raw, reference_bgr_raw = create_realistic_test_images()

    # Preprocess
    source_bgr = cv2.resize(source_bgr_raw, (512, 512))
    reference_bgr = cv2.resize(reference_bgr_raw, (512, 512))
    source_lab = bgr_to_lab(source_bgr)
    reference_lab = bgr_to_lab(reference_bgr)

    # Test all three tiers
    test_reinhard(source_lab, reference_lab, source_bgr)
    test_histogram(source_lab, reference_lab, source_bgr)
    test_neural(source_bgr, reference_bgr)

    # Create comparison
    create_comparison_grid()

    print("\n" + "=" * 60)
    print("  Phase 3 complete - all transfer methods validated!")
    print("  Check test_output/ for visual results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
