# Classification of Skin Diseases

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Notebook-Google%20Colab-F9AB00.svg)](https://colab.research.google.com/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

An image classification system to detect and identify various types of skin conditions using fine-tuned MobileNetV2, built to support preliminary medical diagnosis.

---

## Overview

This project applies transfer learning to classify skin disease images into multiple diagnostic categories. The pipeline includes data validation, stratified splitting (70/15/15), MobileNetV2 fine-tuning, and full evaluation with confusion matrix and classification report.

---

## Repository Structure

```
Classification of skin diseases/
├── classification_of_skin_diseases.py  # Full pipeline: EDA → training → evaluation → export
├── artifacts/
│   ├── skin_disease_mobilenetv2.keras  # Trained model
│   └── class_names.json                # Class label mapping
└── requirements.txt                    # Python dependencies
```

---

## Dataset

- **Source:** Custom skin disease dataset (Google Drive / Kaggle)
- **Format:** Folder-per-class structure
- **Split:** 70% train / 15% validation / 15% test (stratified manual split)
- **Input size:** 224×224 RGB

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/faja27/Intermediate-project.git
cd "Intermediate-project/Classification of skin diseases"
```

### 2. Install dependencies

```bash
pip install tensorflow scikit-learn Pillow numpy matplotlib
```

### 3. Run on Google Colab (recommended)

1. Upload `classification_of_skin_diseases.py` to Colab
2. Mount Google Drive and update `DATASET_DIR` to your dataset path
3. Run all cells in order

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | MobileNetV2 (ImageNet pretrained) |
| Stage 1 | Frozen backbone, train classifier head |
| Stage 2 | Fine-tune top layers of backbone |
| Custom head | GlobalAveragePooling2D → Dropout(0.2) → Dense(num_classes, softmax) |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Input size | 224×224 RGB |

---

## Training Setup

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Seed | 42 (reproducible) |
| Early stopping | patience=4 (monitor val_accuracy) |
| Augmentation | RandomFlip(horizontal), RandomRotation(0.05), RandomZoom(0.10) |
| Evaluation | classification_report + confusion matrix |

---

## Pipeline Steps

| Step | Description |
|---|---|
| 1 | Mount Drive + import libraries |
| 2 | Validate dataset (count images, detect corrupt files) |
| 3 | Manual stratified split (70/15/15) |
| 4 | Build TF datasets with prefetch + cache |
| 5 | Data augmentation pipeline |
| 6 | Build MobileNetV2 classifier |
| 7 | Stage 1 training (frozen backbone) |
| 8 | Stage 2 fine-tuning |
| 9 | Test set evaluation + confusion matrix |
| 10 | Export model + class_names.json |

---

## Computational Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Not required | NVIDIA CUDA / Google Colab GPU |
| Training time | ~1–3 h (CPU) | ~15–30 min (Colab GPU) |

---

## License

This project is licensed under the [MIT License](LICENSE).
