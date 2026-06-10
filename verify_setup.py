"""Quick verification that all dependencies are installed correctly."""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import cv2
print(f"OpenCV: {cv2.__version__}")

import gradio
print(f"Gradio: {gradio.__version__}")

import skimage
print(f"scikit-image: {skimage.__version__}")

import scipy
print(f"SciPy: {scipy.__version__}")

import fastapi
print(f"FastAPI: {fastapi.__version__}")

import numpy
print(f"NumPy: {numpy.__version__}")

from PIL import Image
print(f"Pillow: OK")

print("\n[SUCCESS] All imports successful - Phase 1 setup complete!")
