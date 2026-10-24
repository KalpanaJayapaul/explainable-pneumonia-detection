"""
End-to-end check of the REAL inference pipeline on images from dataset/test.

Run from the project root:
    python ml/verify_pipeline.py                 (5 NORMAL + 5 PNEUMONIA, random)
    python ml/verify_pipeline.py --n 10 --seed 7

It uses the same functions the Django API uses (predictor.inference), so what
you see here is what the web app computes.
"""
import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predictor.inference import (          # noqa: E402
    CLASS_NAMES, IMG_SIZE, MODEL_PATH, get_model, load_rgb,
    risk_from_probability, to_model_input,
)
from predictor.gradcam import make_gradcam  # noqa: E402

TEST = ROOT / "dataset" / "test"
OUT = ROOT / "ml" / "artifacts"
EXT = {".jpg", ".jpeg", ".png"}


def training_style_input(path):
    """Exactly what image_dataset_from_directory does (no PIL involved)."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return np.expand_dims(img.numpy().astype(np.float32), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="images per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- 1. class mapping -------------------------------------------------
    print("=" * 78)
    print("1. CLASS MAPPING")
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST, image_size=IMG_SIZE, batch_size=32, label_mode="binary",
        class_names=CLASS_NAMES, shuffle=False, verbose=0)
    folders = sorted(p.name for p in TEST.iterdir() if p.is_dir())
    print(f"   folders in dataset/test : {folders}")
    print(f"   Keras class_names       : {ds.class_names}  (index 0 -> 0.0, index 1 -> 1.0)")
    assert ds.class_names == ["NORMAL", "PNEUMONIA"], "class order changed!"
    print("   => sigmoid output = P(PNEUMONIA).  OK")

    # ---- 2. model ---------------------------------------------------------
    print("=" * 78)
    print("2. MODEL")
    print(f"   path: {MODEL_PATH}")
    model = get_model()

    # ---- 3. predictions ---------------------------------------------------
    print("=" * 78)
    print(f"3. PREDICTIONS ON {args.n} NORMAL + {args.n} PNEUMONIA TEST IMAGES (seed={args.seed})")
    rng = random.Random(args.seed)
    rows, cam_dir = [], OUT / "verify_gradcam"
    shutil.rmtree(cam_dir, ignore_errors=True)
    cam_dir.mkdir(parents=True)
    tmp_media = OUT / "verify_tmp"

    for actual in CLASS_NAMES:
        files = sorted(p for p in (TEST / actual).iterdir() if p.suffix.lower() in EXT)
        for i, f in enumerate(rng.sample(files, min(args.n, len(files))), 1):
            x = to_model_input(load_rgb(f))
            raw = model.predict(x, verbose=0)
            p = float(raw[0][0])
            idx = int(p >= 0.5)
            pred = CLASS_NAMES[idx]

            x_train = training_style_input(f)                      # preprocessing parity check
            p_train = float(model.predict(x_train, verbose=0)[0][0])

            url, info = make_gradcam(model, load_rgb(f), x, tmp_media, target_class=idx)
            cam_src = tmp_media / "gradcam" / Path(url).name
            cam_dst = cam_dir / f"{actual.lower()}_{i}__pred_{pred.lower()}__{p:.3f}.jpg"
            shutil.move(str(cam_src), cam_dst)

            rows.append(dict(
                image=f.name, actual=actual, predicted=pred, p_pneumonia=p,
                confidence=p if idx else 1 - p, risk=risk_from_probability(p),
                correct=(pred == actual), parity_diff=abs(p - p_train),
                cam_raw_max=info["raw_max"], cam_p_check=info["p_check"],
                cam_md5=hashlib.md5(cam_dst.read_bytes()).hexdigest(),
            ))
    shutil.rmtree(tmp_media, ignore_errors=True)

    hdr = f"{'Image':34s} {'Actual':10s} {'Predicted':10s} {'P(pneu)':>8s} {'Risk':9s} Correct?"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['image'][:34]:34s} {r['actual']:10s} {r['predicted']:10s} "
              f"{r['p_pneumonia']:8.4f} {r['risk']:9s} {'YES' if r['correct'] else 'NO'}")

    # ---- 4. checks --------------------------------------------------------
    print("=" * 78)
    print("4. CHECKS")
    probs = np.array([r["p_pneumonia"] for r in rows])
    p_norm = np.array([r["p_pneumonia"] for r in rows if r["actual"] == "NORMAL"])
    p_pneu = np.array([r["p_pneumonia"] for r in rows if r["actual"] == "PNEUMONIA"])
    acc = np.mean([r["correct"] for r in rows])
    print(f"   accuracy on this sample              : {acc:.0%} ({sum(r['correct'] for r in rows)}/{len(rows)})")
    print(f"   mean P(pneumonia)  NORMAL images     : {p_norm.mean():.4f}")
    print(f"   mean P(pneumonia)  PNEUMONIA images  : {p_pneu.mean():.4f}")
    print(f"   std of all outputs                   : {probs.std():.5f}   (distinct values: {len(set(np.round(probs, 5)))})")
    print(f"   max |web-pipeline - training-pipeline|: {max(r['parity_diff'] for r in rows):.2e}   (should be ~0)")
    print(f"   max |Grad-CAM forward - predict|     : {max(abs(r['cam_p_check'] - r['p_pneumonia']) for r in rows):.2e}   (should be ~0)")
    print(f"   distinct Grad-CAM images             : {len(set(r['cam_md5'] for r in rows))} of {len(rows)}")
    print(f"   Grad-CAM images saved in             : {cam_dir}")

    print("\nVERDICT")
    if probs.std() < 1e-3:
        print("   X  DEGENERATE MODEL: output is (almost) constant for every image.")
        print("      The checkpoint on disk is not a working classifier -> retrain (see instructions).")
    elif p_pneu.mean() - p_norm.mean() < 0.15:
        print("   !  Output varies with the image, but the classes are NOT separated on test data.")
        print("      The pipeline works; the model itself generalises poorly -> run ml/evaluate.py.")
    else:
        print("   OK Output varies per image and PNEUMONIA images score clearly higher than NORMAL.")
        print("      Inference pipeline is working. Check ml/evaluate.py for full-test-set metrics.")

    with open(OUT / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written: {OUT / 'verification_results.csv'}")


if __name__ == "__main__":
    main()