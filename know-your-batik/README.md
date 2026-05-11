# Know Your Batik

A batik pattern classifier using CNN (ResNet-50) with a FastAPI backend and React frontend.

## Dataset
- 28 batik pattern classes
- 2,216 images (.jpg, .jpeg, .png, .webp)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

## Project Structure

```
know-your-batik/
├── data/               # raw + processed splits
├── models/             # saved checkpoints
├── notebooks/          # EDA & experiments
├── src/                # ML pipeline (data, model, train, eval)
├── backend/            # FastAPI app + routes + schemas
├── frontend/           # React app
├── logs/               # training logs
├── outputs/eda/        # EDA plots
├── config.yaml         # central config
└── requirements.txt
```
