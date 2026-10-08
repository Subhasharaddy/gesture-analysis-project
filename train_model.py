"""
Machine Learning Training Pipeline for AI Sign Language Recognition (Pure Laptop).
Loads samples from dataset/<GESTURE>/ folders, trains a Random Forest Classifier,
evaluates performance, and saves model to models/sign_model.pkl.
"""

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import time
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

import config


def load_dataset():
    """Loads all .npy feature vectors from dataset/<GESTURE>/ directories."""
    X_list = []
    y_list = []

    for gesture_name in config.GESTURES:
        folder = config.DATASET_DIR / gesture_name
        if not folder.exists():
            safe_folder = config.DATASET_DIR / gesture_name.replace(" ", "_").replace("/", "_")
            if safe_folder.exists():
                folder = safe_folder
            else:
                continue

        sample_files = list(folder.glob("sample_*.npy"))
        print(f"Loading {len(sample_files):3d} samples for '{gesture_name}'...")

        for s_file in sample_files:
            try:
                vec = np.load(str(s_file))
                X_list.append(vec)
                y_list.append(gesture_name)
            except Exception as e:
                print(f"Warning: Could not read {s_file}: {e}")

    if not X_list:
        return None, None

    return np.array(X_list, dtype=np.float32), np.array(y_list)


def train_sign_classifier():
    print("=" * 65)
    print("AI SIGN LANGUAGE SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"Dataset Directory: {config.DATASET_DIR}")
    print(f"Target Gestures:   {config.GESTURES}")
    print("=" * 65)

    X, y_raw = load_dataset()

    if X is None or len(X) == 0:
        print("[ERROR] No training samples found in dataset/ folders!")
        print("Please run 'python collect_data.py' or 'python seed_dataset.py' first.")
        return False

    print(f"\nTotal Dataset Samples: {len(X)}")
    print(f"Feature Vector Dimensions: {X.shape[1]}")

    # Encode string labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"Encoded Classes: {class_names}")

    # 80/20 Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier (100 Trees, max_depth=16)...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"[SUCCESS] Model trained in {t_train:.2f} seconds.")

    # Model Evaluation: Training and Test Accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS:")
    print("=" * 65)
    print(f"  - Number of Classes:          {len(class_names)}")
    print(f"  - Total Dataset Samples:      {len(X)}")
    print(f"  - Training Samples:           {len(X_train)}")
    print(f"  - Validation/Test Samples:    {len(X_test)}")
    print(f"  - Training Accuracy:          {train_acc * 100:.2f}%")
    print(f"  - Validation/Test Accuracy:   {test_acc * 100:.2f}%")

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  - 5-Fold Cross-Validation:    {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Classification Report
    print("\n" + "=" * 65)
    print("DETAILED CLASSIFICATION REPORT:")
    print("=" * 65)
    report_text = classification_report(y_test, y_test_pred, target_names=class_names, digits=3)
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("=" * 65)
    print("CONFUSION MATRIX (True \\ Pred):")
    print("=" * 65)
    # Header row
    hdr = f"{'':<12}" + "".join([f"{c[:5]:>7}" for c in class_names])
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:11]:<12}" + "".join([f"{val:>7d}" for val in row])
        print(row_str)
    print("=" * 65)

    # Save artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PKL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(config.MODEL_PKL_PATH.name),
        "num_classes": len(class_names),
        "classes": class_names,
        "kannada_translations": {c: config.GESTURES_KANNADA.get(c, "") for c in class_names},
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "confusion_matrix": cm.tolist()
    }

    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("SAVED PRODUCTION MODEL ARTIFACTS:")
    print(f"  - Model:         {config.MODEL_PKL_PATH}")
    print(f"  - Label Encoder: {config.LABEL_ENCODER_PATH}")
    print(f"  - Metadata:      {config.MODEL_METADATA_PATH}")
    print("=" * 65)

    return {
        "success": True,
        "model": model,
        "label_encoder": label_encoder,
        "classes": class_names,
        "num_classes": len(class_names),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_accuracy": float(cv_scores.mean()),
        "confusion_matrix": cm,
        "report": report_text
    }


if __name__ == "__main__":
    train_sign_classifier()

