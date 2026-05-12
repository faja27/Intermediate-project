from pathlib import Path

import torch.nn as nn
import yaml
from torchvision import models


def _load_cfg():
    cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_model(num_classes: int) -> nn.Module:
    cfg = _load_cfg()
    dropout = cfg['model']['dropout']

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for layer_name in ('layer3', 'layer4', 'fc'):
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )

    return model
