from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

MODEL_PATH = _PROJECT_ROOT / "models" / "checkpoint_best.pth"
LABELS_PATH = _PROJECT_ROOT / "models" / "class_labels.pkl"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES: int = _cfg["data"]["num_classes"]
MAX_FILE_SIZE_MB: int = 10
ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
