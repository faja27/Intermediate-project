## Intermediate Projects - Showcase

A collection of intermediate-level projects focusing on Artificial Intelligence, Machine Learning, and Computer Vision. This repository summarizes my exploration in developing intelligent solutions and applying various Deep Learning architectures to specific datasets.

---

### 🚀 Featured Projects

### 1. American Sign Language (ASL) Alphabet Recognition
A real-time ASL alphabet recognition system.
- **Tech Stack:** Python, TensorFlow/Keras, OpenCV, Streamlit.
- **Architecture:** Utilizes a fine-tuned **MobileNetV2** to recognize 29 categories of sign language alphabets with high efficiency for mobile/web platforms.

### 2. Classification of Skin Diseases
An image classification system to detect various types of skin conditions.
- **Tech Stack:** Python, CNN (Convolutional Neural Networks).
- **Objective:** To assist in the early identification of skin conditions using digital image processing to support preliminary medical diagnosis.

### 3. Know Your Batik
A CNN-based batik pattern recognition system with an interactive web interface to help users identify and learn about Indonesian batik motifs.
- **Tech Stack:** Python, TensorFlow/Keras, FastAPI, React.
- **Architecture:** Fine-tuned **ResNet50** trained on a curated dataset of 28 batik classes and 2,128 images sourced from across Indonesia.
- **Features:** Upload any batik image to get top predictions with confidence scores, browse a gallery of all 28 motifs, and explore the cultural background of each pattern.
- **Dataset:** 28 classes including regional motifs such as Jawa_Barat_Megamendung, Kalimantan_Dayak, Papua_Cendrawasih, Yogyakarta_Kawung, and more.
- **Status:** 🔄 In Progress

### 4. XAUBot (Gold Trading Bot)
An ML/AI-based automated trading bot for **XAUUSD (Gold/USD)** on MetaTrader 5, combining breakout strategy with technical indicator signals optimized from 81 parameter combinations over 6 years of historical data (2020–2025).
- **Tech Stack:** Python, MetaTrader5, pandas, numpy.
- **Strategy:** Range Breakout + EMA Trend Filter + RSI Momentum + ATR Volatility Filter with session-based entry (London & New York).
- **Performance:** Win Rate 72.1% | Profit Factor 2.71 | Max Drawdown 4.1% (backtest 2020–2025).
- **Features:** Trailing stop management, dollar-based risk controls, daily circuit breaker, automated trade journal to Excel.

### 5. RealFace — AI Face Authenticity Detector
A binary image classification system that detects whether a face image is a real photograph or an AI-generated (synthetic) face, powered by EfficientNet-B0 fine-tuned on StyleGAN3-generated faces.
- **Tech Stack:** Python, PyTorch, ONNX Runtime, Streamlit.
- **Architecture:** Fine-tuned **EfficientNet-B0** using a two-phase training strategy (frozen backbone → full fine-tune) on 10,000 face images (5,000 real + 5,000 StyleGAN3-generated).
- **Results:** Test Accuracy 90.13% | Precision 90.90% | Recall 89.20% | F1 90.04% | AUC 96.75%
- **Dataset:** [10000 Real vs Fake Faces (StyleGAN3)](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3) — Kaggle
- **Live Demo:** [🚀 Try it here](https://intermediate-project-2sv9h8q6t2prprz4ibeeag.streamlit.app/)
- **Status:** ✅ Completed

---

## 🛠️ Tech Stack & Tools
- **Languages:** Python, SQL
- **AI/ML:** TensorFlow, Keras, PyTorch, Scikit-Learn, ONNX Runtime
- **Architectures:** CNN, MobileNetV2, ResNet50, EfficientNet-B0, LSTM
- **Tools:** Google Colab, Git/GitHub, Streamlit, FastAPI, React, MetaTrader5

---

## 📂 How to Run the Projects
Each project folder has different dependencies. Generally, you can run a project by:
1. Cloning this repository:
   ```bash
   git clone https://github.com/faja27/Intermediate-project.git
   ```
2. Navigate to the project folder and follow the `README.md` inside each subdirectory.
