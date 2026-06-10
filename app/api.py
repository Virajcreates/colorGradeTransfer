"""
FastAPI endpoints for the Color Grade Transfer tool (optional).
Provides a REST API for programmatic access to the transfer pipeline.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from typing import Optional

from core.preprocess import load_and_convert, lab_to_bgr, bgr_to_rgb, rgb_to_bgr
from core.transfer import reinhard_transfer, histogram_transfer, neural_transfer
from core.postprocess import postprocess

app = FastAPI(
    title="Color Grade Transfer API",
    description="Transfer color grading from a reference image to a source image.",
    version="0.1.0",
)


def _read_upload_as_rgb(upload: UploadFile) -> np.ndarray:
    """Read an uploaded file into an RGB uint8 numpy array."""
    contents = upload.file.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    return np.array(pil_image)


@app.post("/transfer", summary="Apply color grade transfer")
async def transfer(
    source: UploadFile = File(..., description="Source image file"),
    reference: UploadFile = File(..., description="Reference image file"),
    method: str = Form("histogram", description="Transfer method: statistical, histogram, or neural"),
    strength: float = Form(0.8, description="Blend strength (0.0–1.0)"),
    preserve_luminance: bool = Form(True, description="Preserve source luminance"),
    sharpen: bool = Form(True, description="Apply post-transfer sharpening"),
):
    """
    Upload a source and reference image, receive the color-graded result.
    """
    try:
        source_rgb = _read_upload_as_rgb(source)
        reference_rgb = _read_upload_as_rgb(reference)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read images: {e}")

    # Convert to LAB
    source_lab = load_and_convert(source_rgb, size=(512, 512))
    reference_lab = load_and_convert(reference_rgb, size=(512, 512))

    method_lower = method.lower().strip()

    if method_lower == "statistical":
        transferred_lab = reinhard_transfer(source_lab, reference_lab)
    elif method_lower == "histogram":
        transferred_lab = histogram_transfer(source_lab, reference_lab)
    elif method_lower == "neural":
        source_resized = cv2.resize(source_rgb, (512, 512))
        reference_resized = cv2.resize(reference_rgb, (512, 512))
        result_rgb = neural_transfer(source_resized, reference_resized)
        result_bgr = rgb_to_bgr(result_rgb)
        transferred_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:
        raise HTTPException(status_code=400,
                            detail=f"Unknown method: {method}. "
                                   f"Use 'statistical', 'histogram', or 'neural'.")

    # Postprocess
    result_bgr = postprocess(
        source_lab=source_lab,
        transferred_lab=transferred_lab,
        strength=strength,
        preserve_lum=preserve_luminance,
        sharpen=sharpen,
    )

    # Convert to PNG and stream back
    result_rgb = bgr_to_rgb(result_bgr)
    pil_result = Image.fromarray(result_rgb)
    buf = BytesIO()
    pil_result.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
