# RealFace — AI Face Authenticity Detector

## Overview
RealFace is a binary image classifier that detects whether a face photo is a real photograph or an AI-generated (synthetic) image. It is built on a fine-tuned EfficientNet-B0 model and served through a lightweight Streamlit web app.

## Tech Stack
- **Model**: EfficientNet-B0 (fine-tuned, exported to ONNX)
- **Training**: Google Colab + PyTorch
- **Dataset**: Real vs Fake Faces StyleGAN3 (Kaggle) — 10 000 samples (5 000 real · 5 000 fake)
- **Web**: Streamlit
- **Deploy**: Streamlit Community Cloud

## Project Structure
```
realface/
├── app.py                      # Streamlit web app
├── requirements.txt
├── README.md
├── model/
│   └── realface.onnx           # Exported ONNX model
├── utils/
│   └── preprocess.py           # Shared preprocessing function
└── notebook/
    └── realface_colab.ipynb    # Full training pipeline (Colab)
```

## How to Run Locally
1. **Clone the repo and enter the directory**
   ```bash
   git clone <repo-url>
   cd realface
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place the model file**
   Copy `realface.onnx` into the `model/` folder.
4. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## Training
The full training pipeline — data download, EDA, preprocessing, model training, evaluation, and ONNX export — is available in [`notebook/realface_colab.ipynb`](notebook/realface_colab.ipynb). It is designed to run end-to-end on Google Colab with a GPU runtime.

## Results
Model achieved XX% accuracy on the test set.
