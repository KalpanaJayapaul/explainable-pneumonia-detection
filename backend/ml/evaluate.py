from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MODEL = ROOT / "ml" / "artifacts" / "pneumonia_resnet50.keras"

def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET / "test",
        image_size=(224,224),
        batch_size=32,
        label_mode="binary",
        class_names=["NORMAL","PNEUMONIA"],
        shuffle=False
    )
    model = tf.keras.models.load_model(MODEL)
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    probs = model.predict(test_ds, verbose=0).ravel()
    y_pred = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("F1       :", f1_score(y_true, y_pred, zero_division=0))
    print("Specificity:", specificity)
    print("ROC-AUC  :", roc_auc_score(y_true, probs))
    print("\\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=["NORMAL","PNEUMONIA"], zero_division=0
    ))
    print("Confusion matrix:\\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
