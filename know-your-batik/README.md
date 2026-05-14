# Know Your Batik

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB.svg)](https://react.dev/)
[![Status](https://img.shields.io/badge/Status-Beta%20%2F%20Demo%20Ready-blue.svg)]()

A CNN-based Indonesian batik pattern recognition system with an interactive web interface for identifying and learning about batik motifs from across the archipelago.

---

## Overview

This project applies transfer learning to classify 28 distinct Indonesian batik patterns from a curated image dataset. The system includes:

- **ResNet-50** fine-tuned on 2,128 cleaned images across 28 classes (after corrupt/duplicate removal)
- **FastAPI** backend serving the trained model via REST API
- **React** frontend with drag-and-drop image upload, top-5 confidence visualization, and a batik knowledge gallery

The 28 classes span batik motifs from across Indonesia — from Yogyakarta_Kawung and Solo_Parang (Javanese court batik) to Papua_Cendrawasih and Kalimantan_Dayak (eastern archipelago motifs).

---

## Repository Structure

```
know-your-batik/
├── data/
│   ├── raw/                        # Original 28-class dataset (not tracked by git)
│   └── processed/
│       ├── train/                  # 1,701 images (80%)
│       ├── val/                    # 213 images (10%)
│       └── test/                   # 213 images (10%)
├── models/
│   ├── class_labels.pkl            # Class index ↔ name mapping
│   ├── training_history.json       # Loss & accuracy per epoch
│   └── checkpoint_best.pth        # Best epoch checkpoint — NOT tracked by git (270 MB)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb # Cleaning, splitting, DataLoader
│   ├── 03_model_training.ipynb     # Training & validation loop
│   └── 04_model_evaluation.ipynb  # Test set evaluation & confusion matrix
├── src/
│   ├── data_loader.py              # BatikDataset + get_dataloaders()
│   ├── preprocessor.py             # get_transforms() + get_class_weights()
│   ├── model.py                    # get_model() — ResNet50 with custom head
│   ├── trainer.py                  # train_one_epoch() + validate()
│   ├── evaluator.py                # Metrics: accuracy, F1, confusion matrix
│   └── utils.py                    # Helper functions
├── backend/
│   ├── app.py                      # FastAPI application entry point
│   ├── models/
│   │   └── predictor.py            # BatikPredictor class + BATIK_INFO dict
│   ├── routes/
│   │   ├── predict.py              # POST /predict
│   │   └── batik_info.py           # GET /batik/{class_name}, GET /classes
│   └── schemas/
│       └── request_response.py     # Pydantic request/response models
├── frontend/                       # React + Vite + TypeScript web interface
│   ├── src/
│   │   ├── api/
│   │   │   └── batikApi.ts         # API client: predictBatik, getBatikInfo, getAllClasses
│   │   ├── components/
│   │   │   ├── BatikInfoCard.tsx
│   │   │   ├── Navbar.tsx
│   │   │   └── Footer.tsx
│   │   └── pages/
│   │       ├── Home.tsx            # Hero, stats, featured batik
│   │       ├── Classifier.tsx      # Drag-drop upload + top-5 confidence chart
│   │       ├── Gallery.tsx         # Grid + search + detail modal
│   │       └── Learn.tsx           # Educational sections
│   ├── index.html
│   ├── vite.config.ts              # Proxy /api → localhost:8000
│   ├── tailwind.config.js          # Custom batik color palette
│   └── package.json
├── outputs/
│   ├── eda/                        # EDA plots (class distribution, dimensions, etc.)
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── classification_report.txt
│   └── training_curves.png
├── config.yaml                     # Central configuration
├── requirements.txt                # Python dependencies
└── README.md
```

> **Note:** `models/checkpoint_best.pth` (270 MB) is excluded from git via `.gitignore`.
> Download the checkpoint separately and place it at `models/checkpoint_best.pth` before running the backend.

---

## Dataset

### Origin

The dataset used in this project is a curated and modified version of **Batik-Indonesia** created by [Muhammad Salman Al Faridzi](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia), available on HuggingFace and Kaggle under the **Apache 2.0 License**.

The original dataset is a curated collection of batik designs from various regions across Indonesia. Batik is an ancient textile art form that utilizes a wax-resist dyeing technique, resulting in intricate patterns and motifs that reflect cultural identities and traditions across the archipelago. Images were collected from various cultural institutions, museums, and local artisans across Indonesia — including direct collaboration with craftspeople to obtain authentic patterns and digitization of physical batik samples into high-quality image files.

### Modifications from Original

The original dataset contained **38 classes and ~2,599 images**. This project uses a data-centric audited version to improve CNN training quality:

| Change | Detail |
|---|---|
| 10 classes removed | Geographic umbrellas too broad (Bali, Pekalongan), dyeing techniques not spatial motifs (Celup/Gentongan), extreme intra-class variance (Ciamis, Garutan), class overlap (Betawi, Sidomukti, Keraton), or insufficient images (Aceh) |
| 1 class replaced | Priangan → Priangan_Merak_Ngibing (specific subtype with consistent, repeatable geometric structure suitable for CNN classification) |
| 4 classes expanded | Sogan, Lasem, Ceplok, Priangan_Merak_Ngibing (additional images sourced and manually verified) |
| After cleaning | 2,128 valid images (corrupt and duplicate images removed) |

### Dataset Statistics

| Metric | Value |
|---|---|
| Total classes | 28 |
| Total images (after cleaning) | 2,128 |
| Train / Val / Test | 1,701 / 213 / 213 (stratified 80/10/10, seed=42) |
| Largest class | Jawa_Barat_Megamendung — 202 images |
| Smallest class | Priangan_Merak_Ngibing — 34 images |
| Imbalance ratio | ~6× → handled via class-weighted CrossEntropyLoss |
| Image formats | .jpg, .jpeg, .png, .webp |

<details>
<summary>Full class list (28 classes)</summary>

Bali_Barong, Bali_Merak, Ceplok, Corak_Insang, Ikat_Celup, Jakarta_Ondel_Ondel,
Jawa_Barat_Megamendung, Jawa_Timur_Pring, Kalimantan_Dayak, Lampung_Gajah, Lasem,
Madura_Mataketeran, Maluku_Pala, NTB_Lumbung, Papua_Asmat, Papua_Cendrawasih,
Papua_Tifa, Priangan_Merak_Ngibing, Sekar, Sidoluhur, Sogan, Solo_Parang,
Sulawesi_Selatan_Lontara, Sumatera_Barat_Rumah_Minang, Sumatera_Utara_Boraspati,
Tambal, Yogyakarta_Kawung, Yogyakarta_Parang

</details>

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (LTS recommended)
- `checkpoint_best.pth` placed at `models/checkpoint_best.pth`

### 1. Clone the repository

```bash
git clone https://github.com/faja27/Intermediate-project.git
cd Intermediate-project/know-your-batik
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> For GPU training, replace the last command with the CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Run the backend

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)
Expected: `{"status":"ok","model_loaded":true,"num_classes":28}`

### 5. Run the frontend

```bash
cd frontend

# If installing for the first time (slow registry? use mirror):
npm install --registry=https://registry.npmmirror.com

npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### 6. Run the ML pipeline (optional — notebooks in order)

```bash
jupyter notebook
```

| Notebook | Description | Est. Runtime |
|---|---|---|
| `01_eda.ipynb` | Data quality check, class distribution, visualizations | ~5 min |
| `02_data_preprocessing.ipynb` | Clean, stratified split, build DataLoader | ~10 min |
| `03_model_training.ipynb` | Train ResNet50 (50 epochs, early stopping) | ~9 h (CPU) |
| `04_model_evaluation.ipynb` | Test accuracy, confusion matrix, per-class F1 | ~5 min |

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | ResNet-50 (ImageNet pretrained) |
| Frozen layers | layer1, layer2 |
| Trainable layers | layer3, layer4, fc |
| Custom head | Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→28) |
| Total parameters | 24,571,484 |
| Trainable parameters | 23,126,556 |
| Optimizer | AdamW (lr=0.0001, weight_decay=0.0001) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (class-weighted) |
| Input size | 224×224 RGB |

---

## Training Setup

| Parameter | Value |
|---|---|
| Epochs | 50 (+ early stopping) |
| Actual epochs run | 26 (early stopping triggered) |
| Best checkpoint | Epoch 16 |
| Batch size | 32 |
| Early stopping patience | 10 epochs |
| Training device | CPU |
| Training time | ~9 hours |
| Augmentation | RandomRotation(15°), RandomHorizontalFlip, ColorJitter(0.2, 0.2, 0.1, 0.05) |
| Normalization | ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |
| Class imbalance handling | sklearn compute_class_weight('balanced') |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **85.92%** |
| Macro F1 Score | **0.8321** |
| Weighted F1 Score | **0.8559** |
| Best Val Accuracy | 0.9108 (Epoch 16) |

### Per-class highlights

| Category | Classes | F1 Score |
|---|---|---|
| Perfect score | Lampung_Gajah, Madura_Mataketeran, NTB_Lumbung, Papua_Asmat, Papua_Tifa, Sumatera_Barat_Rumah_Minang, Sumatera_Utara_Boraspati | 1.00 |
| Most challenging | Priangan_Merak_Ngibing | 0.40 |
| Most confused pair | Yogyakarta_Parang ↔ Solo_Parang | 7 misclassifications |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — model load status |
| `POST` | `/predict` | Upload image → return top-5 predictions with confidence scores |
| `GET` | `/batik/{class_name}` | Return batik info (origin, history, visual characteristics) |
| `GET` | `/classes` | List all 28 classes |

---

## Computational Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| Node.js | 18 LTS | 20 LTS |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 4 GB |
| GPU | Not required | NVIDIA CUDA-compatible |
| Training time | ~9 h (CPU) | ~30 min (GPU) |

All development was done on Windows 10, without a dedicated GPU.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

The dataset is derived from the [Batik-Indonesia dataset by Muhammad Salman Al Faridzi](https://www.kaggle.com/datasets/muhammadsalmanalfaridzi/batik-indonesia), also released under Apache 2.0.
