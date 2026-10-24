"""
Run from the project root:   python ml/diagnose.py
Optional:                    python ml/diagnose.py path/to/xray1.jpg path/to/xray2.jpg
"""
from pathlib import Path
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MODEL = ROOT / "ml" / "artifacts" / "pneumonia_resnet50.keras"
IMG = (224, 224)
EXT = {".jpg", ".jpeg", ".png"}

model = tf.keras.models.load_model(MODEL)


def prob_web(path):
    """Exactly what inference.py does."""
    arr = np.asarray(Image.open(path).convert("RGB").resize(IMG), dtype=np.float32)
    return float(model.predict(arr[None], verbose=0)[0][0])


def prob_keras(path):
    """Exactly how training loaded images."""
    arr = tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=IMG))
    return float(model.predict(arr[None], verbose=0)[0][0])


def sample(split, cls, n=15):
    files = [p for p in (DATASET / split / cls).iterdir() if p.suffix.lower() in EXT]
    random.seed(0)
    return random.sample(files, min(n, len(files)))


print("\n=== 1. Does the model separate classes on DATASET images? ===")
for split in ("train", "test"):
    for cls in ("NORMAL", "PNEUMONIA"):
        probs = [prob_web(p) for p in sample(split, cls)]
        print(f"{split:5s} {cls:9s} mean P(pneumonia)={np.mean(probs):.3f}  "
              f"min={min(probs):.3f} max={max(probs):.3f}")

print("\n=== 2. Web preprocessing vs training preprocessing (should match) ===")
for p in sample("test", "PNEUMONIA", 5):
    print(f"{p.name[:30]:30s} web={prob_web(p):.4f}  keras={prob_keras(p):.4f}")

if len(sys.argv) > 1:
    print("\n=== 3. Your own images ===")
    for f in sys.argv[1:]:
        print(f"{Path(f).name:30s} P(pneumonia)={prob_web(f):.4f}")