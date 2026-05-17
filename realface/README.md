# RealFace — AI Face Authenticity Detector

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Web-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED.svg)](https://onnx.ai/)
[![Status](https://img.shields.io/badge/Status-Live%20Demo-brightgreen.svg)]()

A binary image classification system that detects whether a face image is a **real photograph** or an **AI-generated (synthetic) face**, powered by EfficientNet-B0 fine-tuned on StyleGAN3-generated faces.

🚀 **[Try the Live Demo](https://intermediate-project-2sv9h8q6t2prprz4ibeeag.streamlit.app/)**

📝 **[Read the Build Thread]()**  <!-- LinkedIn post link — coming soon -->

---

## Overview

This project applies transfer learning to distinguish real human faces from AI-generated faces produced by NVIDIA's StyleGAN3 model. The system includes:

- **EfficientNet-B0** fine-tuned on 10,000 face images (5,000 real + 5,000 synthetic)
- **Two-phase fine-tuning** strategy: frozen backbone → full fine-tune
- **ONNX export** for lightweight, framework-independent inference
- **Streamlit** web app deployed on Streamlit Community Cloud

---

## Repository Structure

```
realface/
├── assets/
│   ├── confusion_matrix.png        # Confusion matrix on test set
│   ├── roc_curve.png               # ROC curve with AUC score
│   └── training_history.png        # Train vs val loss & accuracy
├── model/
│   └── realface.onnx               # Exported ONNX model (~15 MB)
├── notebook/
│   └── realface_colab.ipynb        # Full training pipeline (Google Colab)
├── utils/
│   └── preprocess.py               # preprocess_image() — identical to training transforms
├── app.py                          # Streamlit web application
└── requirements.txt                # Python dependencies
```

---

## Dataset

### Origin

[10000 Real vs Fake Faces (StyleGAN3)](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3) by troykueh on Kaggle.

The dataset provides a clean, standardized, and perfectly balanced baseline for training deepfake detection models. Fake faces were generated using NVIDIA's StyleGAN3, one of the most photorealistic face synthesis models available.

### Sampling Strategy

The original dataset contains 10,000 real and 10,000 fake images. This project samples **5,000 real + 5,000 fake (10,000 total)** for efficiency on Google Colab free tier.

### Dataset Statistics

| Metric | Value |
|---|---|
| Total samples used | 10,000 |
| Real faces | 5,000 |
| Fake faces (StyleGAN3) | 5,000 |
| Train / Val / Test | 7,000 / 1,500 / 1,500 (70/15/15, stratified, seed=42) |
| Image input size | 224×224 RGB |
| License | CC BY-NC-SA 4.0 |

---

## Quick Start

### Prerequisites

- Python 3.10+
- `realface.onnx` already included in `model/`

### 1. Clone the repository

```bash
git clone https://github.com/faja27/Intermediate-project.git
cd Intermediate-project/realface
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | EfficientNet-B0 (ImageNet pretrained) |
| Fine-tuning strategy | Two-phase: classifier only → full model |
| Custom head | Dropout(0.2) → Linear(1280→1) |
| Total parameters | 4,008,829 |
| Optimizer | AdamW |
| Loss function | BCEWithLogitsLoss |
| Scheduler | CosineAnnealingLR (T_max=10) |
| Input size | 224×224 RGB |
| Export format | ONNX (opset 18, ~15 MB) |

---

## Training Setup

| Parameter | Value |
|---|---|
| Total epochs | 10 |
| Phase 1 (epoch 1–3) | Freeze backbone, train head only, lr=1e-3 |
| Phase 2 (epoch 4–10) | Unfreeze all layers, lr=1e-4 |
| Batch size | 32 |
| Training device | NVIDIA T4 (Google Colab free) |
| Augmentation | RandomHorizontalFlip, RandomRotation(10°), ColorJitter |
| Normalization | ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| Checkpoint saving | Best val accuracy only, saved to Google Drive |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **90.13%** |
| Precision | **90.90%** |
| Recall | **89.20%** |
| F1-Score | **90.04%** |
| AUC (ROC) | **96.75%** |

### Confusion Matrix (Test Set — 1,500 samples)

|  | Predicted Fake | Predicted Real |
|---|---|---|
| **Actual Fake** | 683 (TN) | 67 (FP) |
| **Actual Real** | 81 (FN) | 669 (TP) |

### Training History

| Epoch | Train Acc | Val Acc | Notes |
|---|---|---|---|
| 1–3 | 63–68% | 70–73% | Phase 1: classifier only |
| 4 | 74.5% | 81.9% | Phase 2 starts — backbone unfrozen |
| 7 | 90.2% | 89.1% | Consistent improvement |
| 10 | **93.4%** | **90.4%** | Best checkpoint saved |

### Error Analysis

False Positives (fake detected as real) tend to be **high-quality StyleGAN3 outputs** with natural lighting, realistic skin texture, and frontal pose — the hardest cases even for human observers. This reflects the known upper bound of detection difficulty for state-of-the-art generative models.

---

## Inference Pipeline

```
Input image (any size)
        ↓
Resize to 224×224 → CenterCrop(224)
        ↓
Normalize (ImageNet mean/std)
        ↓
ONNX Runtime inference (CPU)
        ↓
Sigmoid → confidence score
        ↓
Output: REAL / FAKE + confidence %
```

Preprocessing is identical between training and inference, implemented in `utils/preprocess.py`.

---

## Computational Requirements

| Component | Value |
|---|---|
| Training platform | Google Colab (free tier) |
| Training GPU | NVIDIA T4 |
| Training time | ~90 minutes (10 epochs, 10k samples) |
| Inference | CPU only (ONNX Runtime) |
| Model size | ~15 MB (ONNX) |
| RAM at inference | < 200 MB |

---

## Limitations

- Trained on StyleGAN3 synthetic faces only — may not generalize well to faces generated by diffusion models (Stable Diffusion, Midjourney, DALL-E)
- Performance may degrade on heavily compressed, cropped, or non-frontal face images
- Dataset size (10k) is intentionally limited for Colab free tier — larger datasets would likely improve recall on hard cases

---

## License

This project is for educational and portfolio purposes.

Dataset: [10000 Real vs Fake Faces (StyleGAN3)](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3) — CC BY-NC-SA 4.0
