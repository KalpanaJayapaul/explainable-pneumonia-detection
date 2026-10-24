"""
Grad-CAM for the trained ResNet50 classifier.

Why the old version produced a flat blue image for NORMAL predictions:
  it always back-propagated the PNEUMONIA probability and then applied
  ReLU.  For an image the model believes is NORMAL, every region pushes the
  pneumonia score DOWN, so after ReLU the whole map is exactly 0 -> uniform
  dark blue (JET colour 0).

Fix: explain the class that was actually predicted.
  predicted PNEUMONIA -> target = +logit   (evidence FOR pneumonia)
  predicted NORMAL    -> target = -logit   (evidence FOR normal)
The logit (pre-sigmoid) is used so the gradient does not vanish when the
sigmoid saturates near 0 or 1.
"""
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

log = logging.getLogger("predictor.gradcam")

GRADCAM_LAYER = "conv5_block3_out"      # last conv block of ResNet50 (7x7x2048)


def make_gradcam(model, rgb, x, media_root, target_class):
    """
    model        : loaded Keras model (contains nested 'resnet50')
    rgb          : the uploaded image as RGB uint8 (H,W,3) - used for the overlay
    x            : float32 (1,224,224,3), 0..255 - the EXACT tensor given to model.predict
    target_class : 0 (NORMAL) or 1 (PNEUMONIA) - the class the model predicted
    returns      : (url, info)
    """
    base_model = model.get_layer("resnet50")
    conv_layer = base_model.get_layer(GRADCAM_LAYER)       # raises if missing

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output],
    )

    base_index = next(i for i, l in enumerate(model.layers) if l is base_model)
    head = model.layers[base_index + 1:]                   # GAP, Dropout, Dense, Dropout, Dense(sigmoid)
    final = head[-1]

    # ResNet50 preprocessing lives inside the model before the base network,
    # so apply it here (on a COPY - Keras may modify numpy input in place).
    tensor = tf.keras.applications.resnet50.preprocess_input(np.array(x, dtype=np.float32, copy=True))

    with tf.GradientTape() as tape:
        conv_out, feat = grad_model(tensor, training=False)
        h = feat
        for layer in head[:-1]:
            h = layer(h, training=False)
        logit = tf.matmul(h, tf.convert_to_tensor(final.kernel)) + tf.convert_to_tensor(final.bias)
        prob = tf.sigmoid(logit)[:, 0]
        score = logit[:, 0] if target_class == 1 else -logit[:, 0]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise ValueError("Grad-CAM gradients could not be calculated.")

    weights = tf.reduce_mean(grads, axis=(1, 2))                          # (1, 2048)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)    # (1, 7, 7)
    cam = tf.maximum(cam, 0)[0].numpy()

    raw_max = float(cam.max())
    cam = cam / raw_max if raw_max > 1e-12 else cam      # all-zero stays zero (reported in `info`)

    h_img, w_img = rgb.shape[:2]
    cam = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)   # BGR
    original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.55, colored, 0.45, 0)

    out_dir = Path(media_root) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    if not cv2.imwrite(str(out_dir / filename), overlay):
        raise ValueError("Failed to save Grad-CAM image.")

    info = {"raw_max": raw_max, "p_check": float(prob.numpy()[0])}
    if raw_max <= 1e-12:
        log.warning("Grad-CAM heatmap is all zeros for target class %d", target_class)
    return "/media/gradcam/" + filename, info