"""
Inference pipeline for the pneumonia classifier.

uploaded file -> RGB uint8 -> tf.image.resize (bilinear, 224x224, float32 0..255)
-> trained model (ResNet50 preprocess_input is INSIDE the model) -> sigmoid output
-> class decision -> Grad-CAM on the SAME tensor.

Nothing in here is hard-coded: label, confidence and risk are all derived
from the value returned by model.predict().
"""
import datetime
import hashlib
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .gradcam import make_gradcam, GRADCAM_LAYER

# ---------------------------------------------------------------- logging
# Prints to the `runserver` console without touching settings.py
log = logging.getLogger("predictor")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# ---------------------------------------------------------------- constants
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "ml" / "artifacts" / "pneumonia_resnet50.keras"
IMG_SIZE = (224, 224)

# train.py passes class_names=["NORMAL","PNEUMONIA"] explicitly, so
# index 0 = NORMAL, index 1 = PNEUMONIA, and the single sigmoid output is
# P(PNEUMONIA).  ml/verify_pipeline.py re-checks this against the dataset folders.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

RISK_NOTE = "Model-Derived Research Category \u2014 Not a Clinically Validated Risk Score."

_model = None
_model_mtime = None
_model_info = {}


# ---------------------------------------------------------------- model loading
def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_and_probe(model):
    """Check that the loaded model has the architecture train.py builds."""
    out_shape = model.output_shape
    in_shape = model.input_shape
    if out_shape[-1] != 1:
        raise RuntimeError(f"Expected 1 sigmoid output, model has output shape {out_shape}")

    base = model.get_layer("resnet50")                 # raises ValueError if missing
    cam_layer = base.get_layer(GRADCAM_LAYER)          # raises ValueError if missing
    last = model.layers[-1]
    act = getattr(getattr(last, "activation", None), "__name__", "?")
    if act != "sigmoid":
        raise RuntimeError(f"Last layer activation is '{act}', expected 'sigmoid'")

    log.info("model input=%s output=%s last_layer=%s(activation=%s) params=%s",
             in_shape, out_shape, last.name, act, f"{model.count_params():,}")
    log.info("resnet50 sub-model found; Grad-CAM layer '%s' present, output shape=%s",
             GRADCAM_LAYER, cam_layer.output.shape)

    # Degenerate-checkpoint probe: a model that outputs the same number for
    # very different inputs is dead (this is what stage 1 of your training did).
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(0, 255, IMG_SIZE[1], dtype=np.float32)[None, :, None], (IMG_SIZE[0], 1, 3))
    probes = np.stack([
        np.zeros((*IMG_SIZE, 3), np.float32),
        np.full((*IMG_SIZE, 3), 255, np.float32),
        rng.uniform(0, 255, (*IMG_SIZE, 3)).astype(np.float32),
        ramp,
    ])
    p = model.predict(probes, verbose=0).ravel()
    log.info("probe outputs (black, white, noise, ramp) = %s", np.round(p, 4).tolist())
    if float(p.max() - p.min()) < 1e-4:
        log.warning("!!! Model returns the SAME value for very different inputs. "
                    "This checkpoint is degenerate (constant output) - retrain it.")


def get_model():
    """Load the model once; reload automatically if the file on disk changes
    (Django's autoreloader does NOT watch .keras files, so without this a
    retrained model is ignored until the server is restarted)."""
    global _model, _model_mtime, _model_info
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run: python ml/train.py")

    mtime = MODEL_PATH.stat().st_mtime
    if _model is None or mtime != _model_mtime:
        log.info("Loading model: %s", MODEL_PATH)
        log.info("  size=%.1f MB  modified=%s", MODEL_PATH.stat().st_size / 1e6,
                 datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"))
        sha = _sha1(MODEL_PATH)
        log.info("  sha1=%s", sha)
        model = tf.keras.models.load_model(MODEL_PATH)
        _validate_and_probe(model)
        _model, _model_mtime = model, mtime
        _model_info = {"file": str(MODEL_PATH), "sha1": sha[:12]}
    return _model


# ---------------------------------------------------------------- preprocessing
def load_rgb(path):
    """Decode an image to an RGB uint8 array.

    JPEG/PNG/BMP/GIF: decoded with tf.io.decode_image - the SAME decoder
    image_dataset_from_directory used in training (bit-identical pixels).
    Anything TensorFlow cannot read (e.g. .webp): decoded with PIL instead.
    """
    try:
        img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
        return img.numpy()
    except Exception:
        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_model_input(rgb):
    """RGB uint8 (H,W,3) -> float32 (1,224,224,3), pixel range 0..255.

    tf.image.resize(bilinear) is exactly what image_dataset_from_directory
    uses in training.  ResNet50 preprocess_input is applied INSIDE the model,
    so it must NOT be applied here (and never multiply by 255 again).
    """
    t = tf.image.resize(rgb, IMG_SIZE, method="bilinear")
    return np.expand_dims(t.numpy().astype(np.float32), 0)


def risk_from_probability(p):
    """Research bands - NOT a clinically validated risk score."""
    if p < 0.30:
        return "LOW"
    if p <= 0.70:
        return "MODERATE"
    return "HIGH"


# ---------------------------------------------------------------- main entry
def predict_xray(path, media_root):
    model = get_model()

    rgb = load_rgb(path)
    x = to_model_input(rgb)

    raw = model.predict(x, verbose=0)                  # shape (1, 1)
    p = float(raw[0][0])                               # P(PNEUMONIA)
    pred_idx = int(p >= 0.5)
    label = CLASS_NAMES[pred_idx]
    confidence = p if pred_idx == 1 else 1.0 - p
    risk = risk_from_probability(p)

    cam_url, cam_info = make_gradcam(model, rgb, x, media_root, target_class=pred_idx)

    log.info("file=%s orig_shape=%s input_shape=%s px[min=%.1f max=%.1f mean=%.1f]",
             Path(path).name, rgb.shape, x.shape, x.min(), x.max(), x.mean())
    log.info("  raw_output=%s p(pneumonia)=%.6f -> %s  confidence=%.4f  risk=%s",
             raw.tolist(), p, label, confidence, risk)
    log.info("  gradcam: target=%s layer=%s heatmap_raw_max=%.6g p_check=%.6f",
             CLASS_NAMES[pred_idx], GRADCAM_LAYER, cam_info["raw_max"], cam_info["p_check"])
    if abs(cam_info["p_check"] - p) > 1e-3:
        log.warning("Grad-CAM forward pass (%.6f) != model.predict (%.6f)!", cam_info["p_check"], p)

    return {
        "prediction": label,
        "pneumonia_probability": round(p, 4),
        "confidence": round(confidence, 4),
        "risk_category": risk,
        "risk_note": RISK_NOTE,
        "gradcam_url": cam_url,
        "gradcam_target": CLASS_NAMES[pred_idx],
        "model_sha1": _model_info.get("sha1"),
        "disclaimer": (
            "Research prototype only. Model-derived probability is not a "
            "clinical diagnosis or validated medical risk score."
        ),
    }