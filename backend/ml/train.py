from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
ARTIFACTS = ROOT / "ml" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 10

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        class_names=CLASS_NAMES,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )


def metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def main():
    for d in ("train", "validation", "test"):
        if not (DATASET / d).exists():
            raise FileNotFoundError(f"Missing folder: {DATASET / d}")

    train_ds = load(DATASET / "train", True)
    val_ds = load(DATASET / "validation", False)

    # Class weights (NORMAL is the minority class)
    y_train = np.concatenate([y.numpy().ravel() for _, y in train_ds]).astype(int)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weight)

    # NOTE: no horizontal flip - it moves the heart to the wrong side
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")

    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="chest_xray")
    x = augmentation(inputs)

    # ---------------------------------------------------------------
    # THE BUG FIX:
    # image_dataset_from_directory already yields pixels in 0..255.
    # The old code did preprocess_input(x * 255.0) -> values up to
    # 65025, which saturated ResNet and made the output ~0.5 always.
    # ---------------------------------------------------------------
    x = tf.keras.applications.resnet50.preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="pneumonia_probability"
    )(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=metrics(),
    )

    # One shared checkpoint so the best model across BOTH stages is kept
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(ARTIFACTS / "pneumonia_resnet50.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1,
    )

    def callbacks():
        return [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=4,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, verbose=1
            ),
        ]

    print("\nSTAGE 1: TRANSFER LEARNING")
    h1 = model.fit(
        train_ds.prefetch(tf.data.AUTOTUNE),
        validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
        epochs=INITIAL_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks(),
    )

    print("\nSTAGE 2: FINE TUNING")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:  # keep BatchNorm frozen
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=metrics(),
    )
    h2 = model.fit(
        train_ds.prefetch(tf.data.AUTOTUNE),
        validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks(),
    )

    model.save(ARTIFACTS / "pneumonia_resnet50_final.keras")

    history = {}
    for h in (h1, h2):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(i) for i in v)
    with open(ARTIFACTS / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTRAINING COMPLETE ->", ARTIFACTS / "pneumonia_resnet50.keras")


if __name__ == "__main__":
    main()