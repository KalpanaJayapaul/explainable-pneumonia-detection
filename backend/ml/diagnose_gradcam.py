"""
Find out why Grad-CAM's forward pass disagrees with model.predict.
Run from project root:  python ml/diagnose_gradcam.py [optional_image_path]
"""
import os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from predictor.inference import get_model, load_rgb, to_model_input

model = get_model()
img = Path(sys.argv[1]) if len(sys.argv) > 1 else next((ROOT/"dataset"/"test"/"NORMAL").iterdir())
x = to_model_input(load_rgb(img))
print("\nimage:", img.name)

base = model.get_layer("resnet50")
i = next(k for k, l in enumerate(model.layers) if l is base)
head = model.layers[i+1:]
print("layers before base:", [l.name for l in model.layers[:i]])
print("layers after  base:", [l.name for l in head])

def run_head(feat):
    h = feat
    for l in head:
        h = l(h, training=False)
    return float(h.numpy()[0][0])

pre = tf.keras.applications.resnet50.preprocess_input
a = float(model.predict(x, verbose=0)[0][0])
b = run_head(base(pre(np.array(x, copy=True)), training=False))     # what Grad-CAM does
c = run_head(base(np.array(x, copy=True), training=False))          # WITHOUT preprocess
d = float(model(x, training=False).numpy()[0][0])
print(f"\n(a) model.predict(x)                  = {a:.6f}")
print(f"(d) model(x, training=False)          = {d:.6f}")
print(f"(b) base(preprocess(x)) + head        = {b:.6f}   <- Grad-CAM path")
print(f"(c) base(x) + head (no preprocess)    = {c:.6f}")
print("\nreading: if (a)==(c) the saved model has NO working preprocess_input inside;")
print("         if (a)==(b) the paths agree;  if (a)!=(b) and (a)!=(c) something else differs.")