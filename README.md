# Color Grade Transfer 🎨🎭

An advanced, high-precision color grade transfer engine designed to extract color palettes from reference images and apply them to target source images. Features multiple color transfer algorithms, mask-guided regional grading, batch processing, and 3D LUT export (.cube) for professional editing workflows.

---

## 🌟 Key Features

*   **Four Tiers of Color Transfer Engines**:
    *   **Statistical (Reinhard et al.)**: Super fast ($~10\text{ ms}$) mean and standard deviation matching in LAB space.
    *   **Histogram Matching**: CDF-based distribution alignment across individual color channels ($~50\text{ ms}$).
    *   **Multi-Scale**: Advanced histogram matching across multiple resolution scales to prevent banding and color splotches ($~200\text{ ms}$).
    *   **MKL Optimal Transport (Tier 4)**: Captures cross-channel color correlation using Monge-Kantorovitch Linear transport theory. Prevents Reinhard color cast flaws ($~50\text{ ms}$).
    *   **Neural Style Transfer**: gatys-style VGG19 optimization for texture/style reference transfer ($~11\text{ s}$ on CUDA).
*   **Regional Masking & Auto-Detection**:
    *   **Sky Auto-Detect**: Flood-fill and position texture heuristics to segment any sky.
    *   **Foreground Auto-Detect**: GrabCut segmentation with center-prior.
    *   **Luminance Masks**: Highlights, Midtones, and Shadows targeted adjustments.
    *   **Custom Masks**: Upload custom black-and-white alpha maps.
*   **3D LUT Export (.cube)**: Programmatically generate and download standard 3D Lookup Tables ($17^3$, $25^3$, $33^3$, $65^3$) for DaVinci Resolve, Adobe Premiere, Photoshop, Lightroom, and Final Cut Pro.
*   **Batch Processing**: Grade multiple files at once and output directly to local disk.
*   **Modern Web UI**: Interactive dashboard built with Gradio.

---

## 🛠️ Technology Stack

*   **Logic**: Python 3.8+
*   **Deep Learning (Neural Engine)**: PyTorch & Torchvision (VGG19)
*   **Image Processing**: OpenCV (`opencv-python`), `scikit-image`, `Pillow`, `SciPy`
*   **UI Framework**: Gradio

---

## 📂 Project Architecture

```directory
colorGradeTransfer/
├── app/                      # Application Layer
│   ├── api.py                # FastAPI REST API endpoints
│   └── ui.py                 # Gradio Web UI Layout
├── core/                     # Color Grading Core Engine
│   ├── extractor.py          # VGG19 feature extraction & Gram matrices
│   ├── lut_export.py         # 3D LUT lattice builder & cube exporter
│   ├── masking.py            # Sky, Foreground, and Luminance masking
│   ├── postprocess.py        # Denoise, sharpen, and saturation capping
│   ├── preprocess.py         # Aspect ratio resizing & LAB color conversions
│   └── transfer.py           # Reinhard, Histogram, Multi-Scale, MKL, Neural methods
├── examples/                 # Built-in reference and source images
├── utils/                    # Helper functions (color metrics, stats)
├── test_preprocess.py        # Phase 2 verification script
├── test_transfer.py          # Phase 3 transfer pipeline verification
└── verify_setup.py           # Dependency installation check
```

---

## 🚀 Setup & Installation

### Step 1: Install Dependencies
Ensure you have a virtual environment set up and active, then run:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
Verify CUDA, PyTorch, OpenCV, and all libraries are correctly imported:

```bash
python verify_setup.py
```

### Step 3: Run Verification Test Suites
Verify preprocessing and transfer engines are functioning correctly:

```bash
python test_preprocess.py
python test_transfer.py
```

---

## 🖥️ Launching the Web UI

To boot up the interactive Gradio web application locally:

```bash
python app/ui.py
```
Open `http://localhost:7860` in your web browser to access the interface.

---

## 📦 API Development

To run the FastAPI backend server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
API Documentation will be accessible at `http://localhost:8000/docs`.

---

## 📄 References & Background
*   Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001). *Color transfer between images*. IEEE Computer Graphics and Applications.
*   Pitié, F., Dahyot, R., Kelly, F., & Kokaram, A. (2005). *N-dimensional probability density function transfer and its application to color transfer*. ICCV.
*   Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks*. CVPR.
