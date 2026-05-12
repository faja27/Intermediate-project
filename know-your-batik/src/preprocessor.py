from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms


def _load_cfg():
    cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_transforms(split: str) -> transforms.Compose:
    cfg = _load_cfg()
    size = cfg['data']['image_size']
    mean = cfg['data']['mean']
    std  = cfg['data']['std']
    aug  = cfg['augmentation']

    normalize = transforms.Normalize(mean=mean, std=std)

    if split == 'train':
        return transforms.Compose([
            transforms.RandomRotation(aug['rotation_degrees']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05,
            ),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])

    # val / test — deterministic only
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])


def get_class_weights(processed_path) -> torch.Tensor:
    train_dir = Path(processed_path) / 'train'
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    labels = []
    for cls in classes:
        n = len(list((train_dir / cls).glob('*')))
        labels.extend([cls] * n)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float32)
