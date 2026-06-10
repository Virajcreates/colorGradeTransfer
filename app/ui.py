"""
Gradio UI for the Color Grade Transfer tool.
Phase 7: LUT Export, Mask-Guided Transfer, Batch Processing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import cv2
import time
import tempfile
from core.preprocess import (
    load_and_convert, bgr_to_rgb, lab_to_bgr, bgr_to_lab
)
from core.transfer import (
    reinhard_transfer, histogram_transfer,
    multiscale_transfer, mkl_transfer
)
from core.postprocess import postprocess
from core.masking import (
    create_sky_mask, create_foreground_mask,
    create_luminance_mask, load_custom_mask,
    apply_masked_transfer, create_mask_preview
)
from core.lut_export import (
    generate_lut_from_transfer, export_cube_lut
)


TRANSFER_METHODS = {
    "Statistical (Fast)": reinhard_transfer,
    "Histogram (Balanced)": histogram_transfer,
    "Multi-Scale": multiscale_transfer,
    "MKL Optimal (Best)": mkl_transfer,
}


# ====================================================================
# Core grade function
# ====================================================================

def grade(source_img, reference_img, strength, method,
          preserve_luminance, sharpen, apply_denoise,
          progress=gr.Progress()):
    """Main grading function — single image."""
    if source_img is None or reference_img is None:
        raise gr.Error("Please upload both a source and reference image.")

    start_time = time.time()
    progress(0, desc="Preprocessing images...")

    source_lab = load_and_convert(source_img, size=(512, 512))
    reference_lab = load_and_convert(reference_img, size=(512, 512))

    progress(0.1, desc=f"Applying {method}...")
    transfer_fn = TRANSFER_METHODS.get(method)
    if transfer_fn is None:
        raise gr.Error(f"Unknown method: {method}")

    transferred_lab = transfer_fn(source_lab, reference_lab)

    progress(0.7, desc="Postprocessing...")
    result_bgr = postprocess(
        source_lab=source_lab,
        transferred_lab=transferred_lab,
        strength=strength,
        preserve_lum=preserve_luminance,
        sharpen=sharpen,
        apply_denoise=apply_denoise,
    )

    result_rgb = bgr_to_rgb(result_bgr)

    progress(0.95, desc="Creating comparison...")
    source_resized = cv2.resize(source_img, (512, 512))
    comparison = np.hstack([source_resized, result_rgb])

    elapsed = time.time() - start_time
    status = f"Done in {elapsed:.1f}s | {method} | Strength: {strength:.0%}"

    progress(1.0, desc="Complete!")
    return result_rgb, comparison, status


# ====================================================================
# Mask-Guided Transfer
# ====================================================================

def generate_mask(source_img, mask_type, progress=gr.Progress()):
    """Generate an auto mask for preview."""
    if source_img is None:
        raise gr.Error("Please upload a source image first.")

    progress(0.1, desc=f"Detecting {mask_type}...")
    img = cv2.resize(source_img, (512, 512))

    if mask_type == "Sky":
        mask = create_sky_mask(img)
    elif mask_type == "Foreground":
        mask = create_foreground_mask(img)
    elif mask_type == "Highlights":
        mask = create_luminance_mask(img, "highlights")
    elif mask_type == "Midtones":
        mask = create_luminance_mask(img, "midtones")
    elif mask_type == "Shadows":
        mask = create_luminance_mask(img, "shadows")
    else:
        raise gr.Error(f"Unknown mask type: {mask_type}")

    progress(0.8, desc="Creating preview...")
    preview = create_mask_preview(img, mask)

    # Convert mask to displayable image
    mask_display = (mask * 255).astype(np.uint8)
    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)

    progress(1.0, desc="Done!")
    return preview, mask_display, mask


def masked_grade(source_img, reference_img, mask_data,
                 custom_mask_img, mask_type, strength, method,
                 preserve_luminance, feather,
                 progress=gr.Progress()):
    """Apply color transfer with mask."""
    if source_img is None or reference_img is None:
        raise gr.Error("Please upload both source and reference images.")

    start_time = time.time()
    progress(0, desc="Processing...")

    img_resized = cv2.resize(source_img, (512, 512))

    # Get mask
    if custom_mask_img is not None:
        mask = load_custom_mask(custom_mask_img, (512, 512))
    elif mask_data is not None:
        mask = mask_data
    else:
        raise gr.Error("Please generate a mask or upload a custom one.")

    # Ensure mask is correct shape
    if mask.shape[:2] != (512, 512):
        mask = cv2.resize(mask, (512, 512))

    # Get full transfer result
    progress(0.2, desc=f"Applying {method}...")
    source_lab = load_and_convert(source_img, size=(512, 512))
    reference_lab = load_and_convert(reference_img, size=(512, 512))

    transfer_fn = TRANSFER_METHODS.get(method)
    transferred_lab = transfer_fn(source_lab, reference_lab)

    result_bgr = postprocess(
        source_lab=source_lab,
        transferred_lab=transferred_lab,
        strength=strength,
        preserve_lum=preserve_luminance,
        sharpen=True,
        apply_denoise=False,
    )
    result_rgb = bgr_to_rgb(result_bgr)

    # Apply mask blending
    progress(0.8, desc="Applying mask...")
    masked_result = apply_masked_transfer(
        img_resized, result_rgb, mask, feather=feather
    )

    comparison = np.hstack([img_resized, masked_result])
    elapsed = time.time() - start_time
    status = f"Masked transfer done in {elapsed:.1f}s | {mask_type} | {method}"

    progress(1.0, desc="Complete!")
    return masked_result, comparison, status


# ====================================================================
# LUT Export
# ====================================================================

def export_lut(source_img, reference_img, strength, method,
               preserve_luminance, lut_size,
               progress=gr.Progress()):
    """Generate and export a .cube LUT file."""
    if source_img is None or reference_img is None:
        raise gr.Error("Please upload both source and reference images.")

    start_time = time.time()
    progress(0.1, desc="Generating LUT lattice...")

    transfer_fn = TRANSFER_METHODS.get(method)
    if transfer_fn is None:
        raise gr.Error(f"Unknown method: {method}")

    lut_size = int(lut_size)

    progress(0.3, desc=f"Computing {lut_size}³ color mappings...")
    lut_data = generate_lut_from_transfer(
        source_img=source_img,
        reference_img=reference_img,
        transfer_fn=transfer_fn,
        postprocess_fn=postprocess,
        lut_size=lut_size,
        strength=strength,
        preserve_lum=preserve_luminance,
    )

    progress(0.8, desc="Exporting .cube file...")
    method_short = method.split("(")[0].strip().replace(" ", "_").lower()
    filename = f"color_grade_{method_short}_{lut_size}x{lut_size}x{lut_size}.cube"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "output_luts")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    export_cube_lut(
        lut_data=lut_data,
        filepath=filepath,
        lut_size=lut_size,
        title=f"ColorGradeTransfer_{method_short}"
    )

    elapsed = time.time() - start_time
    file_size_kb = os.path.getsize(filepath) / 1024
    status = (f"LUT exported: {filename} ({file_size_kb:.0f} KB) "
              f"in {elapsed:.1f}s | {lut_size}³ = "
              f"{lut_size**3:,} color entries")

    progress(1.0, desc="Export complete!")
    return filepath, status


# ====================================================================
# Batch Processing
# ====================================================================

def batch_process(source_files, reference_img, strength, method,
                  preserve_luminance, progress=gr.Progress()):
    """Process multiple source images with the same reference."""
    if not source_files or reference_img is None:
        raise gr.Error("Please upload source images and a reference image.")

    start_time = time.time()
    transfer_fn = TRANSFER_METHODS.get(method)
    if transfer_fn is None:
        raise gr.Error(f"Unknown method: {method}")

    reference_lab = load_and_convert(reference_img, size=(512, 512))

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "output_batch")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total = len(source_files)

    for i, src_file in enumerate(source_files):
        progress((i + 0.1) / total, desc=f"Processing {i+1}/{total}...")

        # Load source image
        if isinstance(src_file, str):
            src_img = cv2.imread(src_file)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(src_file)
        else:
            src_img = src_file
            basename = f"image_{i+1}.png"

        source_lab = load_and_convert(src_img, size=(512, 512))

        progress((i + 0.4) / total, desc=f"Transferring {i+1}/{total}...")
        transferred_lab = transfer_fn(source_lab, reference_lab)

        result_bgr = postprocess(
            source_lab=source_lab,
            transferred_lab=transferred_lab,
            strength=strength,
            preserve_lum=preserve_luminance,
            sharpen=True,
            apply_denoise=False,
        )
        result_rgb = bgr_to_rgb(result_bgr)

        # Save output
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(output_dir, f"{name}_graded{ext or '.png'}")
        cv2.imwrite(out_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
        results.append(out_path)

    elapsed = time.time() - start_time
    status = (f"Batch complete: {total} images in {elapsed:.1f}s "
              f"({elapsed/total:.1f}s/image) | Saved to output_batch/")

    progress(1.0, desc="Batch complete!")

    # Return the last result as preview and the gallery
    gallery_images = []
    for path in results:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gallery_images.append(img)

    return gallery_images, status


# ====================================================================
# UI Layout
# ====================================================================

def create_ui():
    """Build the full Gradio UI with all Phase 7 features."""

    custom_css = """
    .gradio-container { max-width: 1300px !important; margin: auto !important; }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important; font-weight: 800 !important;
    }
    .sub-header { text-align: center; color: #888; margin-top: 0 !important; }
    #apply-btn, #mask-apply-btn, #lut-export-btn, #batch-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important; font-size: 1.05em !important;
        padding: 12px 32px !important;
    }
    .method-info {
        font-size: 0.85em; color: #999; padding: 8px 12px;
        border-left: 3px solid #667eea; background: rgba(102, 126, 234, 0.05);
    }
    .feature-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.75em; font-weight: 600; margin-left: 6px;
    }
    """

    with gr.Blocks(
        title="Color Grade Transfer",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=custom_css,
    ) as demo:

        gr.HTML("""
            <div style="text-align: center; padding: 20px 0 10px 0;">
                <h1 class="main-header">Color Grade Transfer</h1>
                <p class="sub-header">
                    Transfer color palettes with precision •
                    Export LUTs • Mask-guided regions • Batch processing
                </p>
            </div>
        """)

        # ==============================================================
        # TAB 1: Single Image Transfer
        # ==============================================================
        with gr.Tab("🎨 Transfer"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    source_input = gr.Image(
                        label="Source Image", type="numpy", height=300,
                        sources=["upload", "clipboard"],
                    )
                with gr.Column(scale=1):
                    reference_input = gr.Image(
                        label="Reference Image (color target)",
                        type="numpy", height=300,
                        sources=["upload", "clipboard"],
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    method_radio = gr.Radio(
                        choices=list(TRANSFER_METHODS.keys()),
                        value="MKL Optimal (Best)",
                        label="Transfer Method",
                    )
                    gr.HTML("""<div class="method-info">
                        <b>Statistical</b> ~10ms |
                        <b>Histogram</b> ~50ms |
                        <b>Multi-Scale</b> ~200ms |
                        <b>MKL</b> ~50ms — optimal cross-channel
                    </div>""")
                with gr.Column(scale=1):
                    strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.80, step=0.05,
                        label="Blend Strength",
                    )

            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    preserve_lum = gr.Checkbox(
                        value=True, label="Preserve Luminance")
                    sharpen_check = gr.Checkbox(
                        value=True, label="Apply Sharpening")
                    denoise_check = gr.Checkbox(
                        value=False, label="Apply Denoising")

            transfer_btn = gr.Button(
                "Apply Color Grade", variant="primary",
                size="lg", elem_id="apply-btn")

            with gr.Tab("Result"):
                output_image = gr.Image(
                    label="Graded Image", type="numpy", height=400)
            with gr.Tab("Side-by-Side"):
                comparison_image = gr.Image(
                    label="Source vs. Result", type="numpy", height=400)

            status_text = gr.Textbox(
                label="Status", interactive=False, lines=1)

            transfer_btn.click(
                fn=grade,
                inputs=[source_input, reference_input, strength_slider,
                        method_radio, preserve_lum, sharpen_check,
                        denoise_check],
                outputs=[output_image, comparison_image, status_text],
            )

        # ==============================================================
        # TAB 2: Mask-Guided Transfer
        # ==============================================================
        with gr.Tab("🎭 Mask Transfer"):
            gr.Markdown("""
            ### Mask-Guided Transfer
            Apply the color grade to **specific regions** only.
            Select a region type or upload a custom mask (white = apply, black = keep).
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    mask_source = gr.Image(
                        label="Source Image", type="numpy", height=250,
                        sources=["upload", "clipboard"])
                with gr.Column(scale=1):
                    mask_reference = gr.Image(
                        label="Reference Image", type="numpy", height=250,
                        sources=["upload", "clipboard"])

            with gr.Row():
                with gr.Column(scale=1):
                    mask_type = gr.Radio(
                        choices=["Sky", "Foreground", "Highlights",
                                 "Midtones", "Shadows"],
                        value="Sky", label="Auto-Detect Region")
                    gen_mask_btn = gr.Button(
                        "Generate Mask", variant="secondary")

                with gr.Column(scale=1):
                    custom_mask = gr.Image(
                        label="Or Upload Custom Mask",
                        type="numpy", height=150,
                        sources=["upload", "clipboard"])

            with gr.Row():
                mask_method = gr.Radio(
                    choices=list(TRANSFER_METHODS.keys()),
                    value="MKL Optimal (Best)", label="Method")
                mask_strength = gr.Slider(
                    0, 1, value=0.85, step=0.05, label="Strength")
                mask_feather = gr.Slider(
                    0, 30, value=10, step=1, label="Edge Feather")
                mask_preserve = gr.Checkbox(
                    value=True, label="Preserve Luminance")

            # Hidden state for computed mask
            mask_state = gr.State(None)

            with gr.Row():
                mask_preview = gr.Image(
                    label="Mask Preview", type="numpy", height=250)
                mask_display = gr.Image(
                    label="Mask", type="numpy", height=250)

            mask_apply_btn = gr.Button(
                "Apply Masked Transfer", variant="primary",
                size="lg", elem_id="mask-apply-btn")

            with gr.Tab("Masked Result"):
                mask_result = gr.Image(
                    label="Result", type="numpy", height=400)
            with gr.Tab("Comparison"):
                mask_comparison = gr.Image(
                    label="Source vs. Masked Result",
                    type="numpy", height=400)

            mask_status = gr.Textbox(
                label="Status", interactive=False, lines=1)

            gen_mask_btn.click(
                fn=generate_mask,
                inputs=[mask_source, mask_type],
                outputs=[mask_preview, mask_display, mask_state])

            mask_apply_btn.click(
                fn=masked_grade,
                inputs=[mask_source, mask_reference, mask_state,
                        custom_mask, mask_type, mask_strength,
                        mask_method, mask_preserve, mask_feather],
                outputs=[mask_result, mask_comparison, mask_status])

        # ==============================================================
        # TAB 3: LUT Export
        # ==============================================================
        with gr.Tab("📦 LUT Export"):
            gr.Markdown("""
            ### Export .cube LUT
            Generate a 3D LUT file from your color transfer that can be
            imported into **DaVinci Resolve**, **Premiere Pro**,
            **Photoshop**, **Lightroom**, and more.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    lut_source = gr.Image(
                        label="Source Image", type="numpy", height=250,
                        sources=["upload", "clipboard"])
                with gr.Column(scale=1):
                    lut_reference = gr.Image(
                        label="Reference Image", type="numpy", height=250,
                        sources=["upload", "clipboard"])

            with gr.Row():
                lut_method = gr.Radio(
                    choices=list(TRANSFER_METHODS.keys()),
                    value="MKL Optimal (Best)", label="Method")
                lut_strength = gr.Slider(
                    0, 1, value=0.85, step=0.05, label="Strength")
                lut_preserve = gr.Checkbox(
                    value=True, label="Preserve Luminance")
                lut_size = gr.Radio(
                    choices=["17", "25", "33", "65"],
                    value="33", label="LUT Resolution",
                    info="33 = standard, 65 = high quality")

            lut_export_btn = gr.Button(
                "Export .cube LUT", variant="primary",
                size="lg", elem_id="lut-export-btn")

            lut_file = gr.File(label="Download LUT", type="filepath")
            lut_status = gr.Textbox(
                label="Status", interactive=False, lines=1)

            lut_export_btn.click(
                fn=export_lut,
                inputs=[lut_source, lut_reference, lut_strength,
                        lut_method, lut_preserve, lut_size],
                outputs=[lut_file, lut_status])

        # ==============================================================
        # TAB 4: Batch Processing
        # ==============================================================
        with gr.Tab("📁 Batch"):
            gr.Markdown("""
            ### Batch Processing
            Apply the same color grade to **multiple images** at once.
            Upload your source images and a single reference.
            Results are saved to `output_batch/`.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    batch_sources = gr.File(
                        label="Source Images (upload multiple)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath")
                with gr.Column(scale=1):
                    batch_reference = gr.Image(
                        label="Reference Image", type="numpy", height=250,
                        sources=["upload", "clipboard"])

            with gr.Row():
                batch_method = gr.Radio(
                    choices=list(TRANSFER_METHODS.keys()),
                    value="MKL Optimal (Best)", label="Method")
                batch_strength = gr.Slider(
                    0, 1, value=0.80, step=0.05, label="Strength")
                batch_preserve = gr.Checkbox(
                    value=True, label="Preserve Luminance")

            batch_btn = gr.Button(
                "Process Batch", variant="primary",
                size="lg", elem_id="batch-btn")

            batch_gallery = gr.Gallery(
                label="Results", columns=3, height=400)

            batch_status = gr.Textbox(
                label="Status", interactive=False, lines=1)

            batch_btn.click(
                fn=batch_process,
                inputs=[batch_sources, batch_reference, batch_strength,
                        batch_method, batch_preserve],
                outputs=[batch_gallery, batch_status])

        # ==============================================================
        # Footer
        # ==============================================================
        gr.Markdown("""
        ---
        ### Methods
        | Method | Speed | Quality | Best for |
        |--------|-------|---------|----------|
        | **Statistical** | ~10ms | Good | Quick preview |
        | **Histogram** | ~50ms | Better | General use |
        | **Multi-Scale** | ~200ms | Better+ | Smoother gradients |
        | **MKL Optimal** | ~50ms | Best | Professional color grading |
        """)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
