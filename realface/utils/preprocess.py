import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Return (1, 3, 224, 224) float32 array ready for ONNX input."""
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224)
