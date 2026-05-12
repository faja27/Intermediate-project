import pickle
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.preprocessor import get_transforms


def _load_cfg():
    cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class BatikDataset(Dataset):
    def __init__(self, split_dir: Path, class_to_idx: dict, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(processed_path=None, batch_size=None, num_workers=2):
    cfg = _load_cfg()

    if processed_path is None:
        processed_path = Path(cfg['data']['processed_path'])
    else:
        processed_path = Path(processed_path)

    if batch_size is None:
        batch_size = cfg['model']['batch_size']

    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models_dir.mkdir(exist_ok=True)
    labels_path = models_dir / 'class_labels.pkl'

    train_dir = processed_path / 'train'
    classes   = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    with open(labels_path, 'wb') as f:
        pickle.dump({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, f)

    datasets = {
        split: BatikDataset(
            split_dir=processed_path / split,
            class_to_idx=class_to_idx,
            transform=get_transforms(split),
        )
        for split in ('train', 'val', 'test')
    }

    train_loader = DataLoader(
        datasets['train'], batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets['val'], batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'], batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    print(f'Class mapping saved → {labels_path}')
    print(f'Train: {len(datasets["train"])}  Val: {len(datasets["val"])}  Test: {len(datasets["test"])}')

    return train_loader, val_loader, test_loader
