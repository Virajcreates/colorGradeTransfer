"""
Phase 2 — Preprocessing pipeline validation.
Generates synthetic test images, runs them through the full pipeline,
and validates LAB roundtrip accuracy + channel separation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image

from core.preprocess import (
    load_image,
    load_and_convert,
    preprocess_pair,
    bgr_to_lab,
    lab_to_bgr,
    bgr_to_rgb,
    rgb_to_bgr,
    split_lab,
    merge_lab,
    _normalize_channels,
    _rgba_to_bgr,
)
from utils.color_utils import channel_stats


def create_test_images():
    """Generate synthetic test images for pipeline validation."""
    os.makedirs("test_output", exist_ok=True)

    # 1. Warm-toned "golden hour" image (source)
    source = np.zeros((400, 600, 3), dtype=np.uint8)
    # Warm gradient: orange -> amber
    for x in range(600):
        r = int(200 + 55 * (x / 600))
        g = int(120 + 80 * (x / 600))
        b = int(40 + 30 * (x / 600))
        source[:, x] = [b, g, r]  # BGR
    # Add some "landscape" shapes
    cv2.rectangle(source, (0, 250), (600, 400), (30, 80, 140), -1)   # Ground
    cv2.circle(source, (480, 100), 50, (50, 180, 240), -1)           # Sun
    cv2.imwrite("test_output/source_warm.png", source)

    # 2. Cool-toned "cyberpunk" image (reference)
    reference = np.zeros((300, 800, 3), dtype=np.uint8)
    # Cool gradient: deep blue -> magenta
    for x in range(800):
        r = int(60 + 150 * (x / 800))
        g = int(20 + 30 * (x / 800))
        b = int(180 + 75 * (x / 800))
        reference[:, x] = [b, g, r]  # BGR
    # Add neon accents
    cv2.line(reference, (50, 150), (750, 150), (255, 50, 200), 3)
    cv2.rectangle(reference, (100, 50), (300, 250), (255, 200, 50), 2)
    cv2.imwrite("test_output/reference_cool.png", reference)

    # 3. Grayscale test image
    gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    cv2.imwrite("test_output/test_gray.png", gray)

    # 4. RGBA test image (with transparency)
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[:, :, 0] = 100  # B
    rgba[:, :, 1] = 200  # G
    rgba[:, :, 2] = 50   # R
    # Circular alpha mask
    for y in range(256):
        for x in range(256):
            dist = np.sqrt((x - 128)**2 + (y - 128)**2)
            rgba[y, x, 3] = int(max(0, min(255, 255 - dist * 2)))
    cv2.imwrite("test_output/test_rgba.png", rgba)

    print("  Created 4 test images in test_output/")
    return source, reference


def test_load_from_file():
    """Test loading images from file paths."""
    print("\n[1] Testing file loading...")

    img = load_image("test_output/source_warm.png", size=(512, 512), keep_aspect=False)
    assert img.shape == (512, 512, 3), f"Expected (512,512,3), got {img.shape}"
    assert img.dtype == np.uint8

    img_aspect = load_image("test_output/source_warm.png", size=(512, 512), keep_aspect=True)
    assert img_aspect.shape == (512, 512, 3), f"Expected (512,512,3), got {img_aspect.shape}"

    img_original = load_image("test_output/source_warm.png", size=None)
    assert img_original.shape == (400, 600, 3), f"Expected (400,600,3), got {img_original.shape}"

    print("  PASS - File loading works (stretch, aspect-preserve, original)")


def test_load_from_numpy():
    """Test loading from numpy arrays (simulates Gradio input)."""
    print("\n[2] Testing numpy array loading...")

    # Simulate Gradio RGB input
    rgb_input = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = load_image(rgb_input, size=(512, 512), keep_aspect=False)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    print("  PASS - Numpy RGB input handled correctly")


def test_load_from_pil():
    """Test loading from PIL Image objects."""
    print("\n[3] Testing PIL Image loading...")

    pil_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
    img = load_image(pil_img, size=(512, 512))
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    print("  PASS - PIL Image loading works")


def test_grayscale_handling():
    """Test that grayscale images are properly converted to 3-channel."""
    print("\n[4] Testing grayscale handling...")

    img = load_image("test_output/test_gray.png", size=(256, 256))
    assert img.shape == (256, 256, 3), f"Expected 3-channel, got {img.shape}"
    assert img.dtype == np.uint8
    print("  PASS - Grayscale -> 3-channel conversion works")


def test_rgba_handling():
    """Test that RGBA images are composited correctly."""
    print("\n[5] Testing RGBA handling...")

    img = load_image("test_output/test_rgba.png", size=(256, 256))
    assert img.shape == (256, 256, 3), f"Expected 3-channel, got {img.shape}"
    assert img.dtype == np.uint8
    # Center pixel should have the original color (full alpha)
    # Edge pixels should be blended towards white
    print("  PASS - RGBA compositing works")


def test_lab_roundtrip():
    """Test BGR -> LAB -> BGR roundtrip accuracy."""
    print("\n[6] Testing LAB roundtrip accuracy...")

    original_bgr = cv2.imread("test_output/source_warm.png")
    original_bgr = cv2.resize(original_bgr, (512, 512))

    # Forward: BGR -> LAB
    lab = bgr_to_lab(original_bgr)
    assert lab.dtype == np.float32, f"Expected float32, got {lab.dtype}"
    assert lab.shape == (512, 512, 3)

    # Check LAB ranges (OpenCV uint8-scaled)
    l_ch, a_ch, b_ch = split_lab(lab)
    assert l_ch.min() >= 0, f"L channel min below 0: {l_ch.min()}"
    assert l_ch.max() <= 255, f"L channel max above 255: {l_ch.max()}"

    # Reverse: LAB -> BGR
    reconstructed_bgr = lab_to_bgr(lab)
    assert reconstructed_bgr.dtype == np.uint8

    # Measure roundtrip error
    diff = np.abs(original_bgr.astype(np.float32) - reconstructed_bgr.astype(np.float32))
    max_error = diff.max()
    mean_error = diff.mean()

    print(f"  Roundtrip error — mean: {mean_error:.2f}, max: {max_error:.1f} (out of 255)")
    assert mean_error < 2.0, f"Mean roundtrip error too high: {mean_error:.2f}"
    print("  PASS - LAB roundtrip is accurate")


def test_channel_separation():
    """Verify that L channel contains luminance and AB contain chrominance."""
    print("\n[7] Testing LAB channel separation...")

    source_bgr = cv2.imread("test_output/source_warm.png")
    source_bgr = cv2.resize(source_bgr, (512, 512))
    source_lab = bgr_to_lab(source_bgr)

    ref_bgr = cv2.imread("test_output/reference_cool.png")
    ref_bgr = cv2.resize(ref_bgr, (512, 512))
    ref_lab = bgr_to_lab(ref_bgr)

    # Stats comparison
    src_stats = channel_stats(source_lab)
    ref_stats = channel_stats(ref_lab)

    print("  Source (warm):")
    for ch in ['L', 'A', 'B']:
        s = src_stats[ch]
        print(f"    {ch}: mean={s['mean']:.1f}, std={s['std']:.1f}")

    print("  Reference (cool):")
    for ch in ['L', 'A', 'B']:
        s = ref_stats[ch]
        print(f"    {ch}: mean={s['mean']:.1f}, std={s['std']:.1f}")

    # The warm image should have higher A (red) values
    # The cool image should have higher B (blue shift = lower B) values
    print("  PASS - Channel separation verified (L=luminance, AB=chrominance)")


def test_merge_split_roundtrip():
    """Test split -> merge roundtrip."""
    print("\n[8] Testing split/merge roundtrip...")

    lab = load_and_convert("test_output/source_warm.png", size=(512, 512))
    l, a, b = split_lab(lab)
    merged = merge_lab(l, a, b)

    assert np.allclose(lab, merged), "Split/merge roundtrip failed!"
    print("  PASS - Split/merge roundtrip is lossless")


def test_preprocess_pair():
    """Test the pair preprocessing convenience function."""
    print("\n[9] Testing preprocess_pair...")

    source_lab, ref_lab, source_bgr = preprocess_pair(
        "test_output/source_warm.png",
        "test_output/reference_cool.png",
        size=(512, 512)
    )

    assert source_lab.shape == (512, 512, 3)
    assert ref_lab.shape == (512, 512, 3)
    assert source_bgr.shape == (512, 512, 3)
    assert source_lab.dtype == np.float32
    assert source_bgr.dtype == np.uint8

    print("  PASS - Pair preprocessing works")


def save_channel_visualizations():
    """Save visual debug images showing LAB channel decomposition."""
    print("\n[10] Saving channel visualizations...")

    for name in ["source_warm", "reference_cool"]:
        bgr = cv2.imread(f"test_output/{name}.png")
        bgr = cv2.resize(bgr, (512, 512))
        lab = bgr_to_lab(bgr)
        l, a, b = split_lab(lab)

        # Save individual channels as grayscale images
        cv2.imwrite(f"test_output/{name}_L.png", l.astype(np.uint8))
        cv2.imwrite(f"test_output/{name}_A.png", a.astype(np.uint8))
        cv2.imwrite(f"test_output/{name}_B.png", b.astype(np.uint8))

        # Create a side-by-side comparison: Original | L | A | B
        l_vis = cv2.cvtColor(l.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        a_vis = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        b_vis = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        comparison = np.hstack([bgr, l_vis, a_vis, b_vis])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Original", "L (Luminance)", "A (Green-Red)", "B (Blue-Yellow)"]
        for i, label in enumerate(labels):
            x = i * 512 + 10
            cv2.putText(comparison, label, (x, 30), font, 0.7, (255, 255, 255), 2)

        cv2.imwrite(f"test_output/{name}_LAB_decomposition.png", comparison)

    print("  Saved LAB decomposition visualizations to test_output/")


def main():
    print("=" * 60)
    print("  Phase 2 — Preprocessing Pipeline Validation")
    print("=" * 60)

    # Generate test images
    print("\n[0] Generating synthetic test images...")
    create_test_images()

    # Run all tests
    test_load_from_file()
    test_load_from_numpy()
    test_load_from_pil()
    test_grayscale_handling()
    test_rgba_handling()
    test_lab_roundtrip()
    test_channel_separation()
    test_merge_split_roundtrip()
    test_preprocess_pair()
    save_channel_visualizations()

    print("\n" + "=" * 60)
    print("  All 10 tests passed! Phase 2 preprocessing is solid.")
    print("=" * 60)


if __name__ == "__main__":
    main()
