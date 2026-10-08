"""
Machine Learning Training Pipeline for Sign Language Recognition.
Loads normalized landmark dataset, trains a Random Forest Classifier,
evaluates precision/recall/F1-score, and exports optimized model artifacts.
"""

import os
import sys
import time
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config
from utils.landmark_processor import LandmarkProcessor
from train_model import train_sign_classifier


def train_gesture_classifier():
    """
    Trains the unified 42-gesture Random Forest model using the dataset CSV,
    or falls back to raw .npy sample directories if CSV is not present.
    """
    print("=" * 65)
    print("AI SIGN LANGUAGE SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 65)

    if not config.DATASET_CSV_PATH.exists():
        print(f"[INFO] Dataset CSV not found at: {config.DATASET_CSV_PATH}")
        print("Falling back to dataset/ sample folders...")
        return train_sign_classifier()

    print(f"Loading dataset from: {config.DATASET_CSV_PATH} ...")
    df = pd.read_csv(config.DATASET_CSV_PATH)
    print(f"Total dataset samples: {len(df)}")

    # Check for missing values
    if df.isnull().values.any():
        print("[WARNING] Found missing values in dataset. Dropping null rows...")
        df = df.dropna()

    # Features and labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y_raw = df["label"].values

    print(f"Number of feature dimensions: {X.shape[1]}")
    print("Class distribution:")
    class_counts = df["label"].value_counts()
    for label, count in class_counts.items():
        print(f"  - {label:<15}: {count} samples")

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)
    print(f"\nEncoded classes ({len(class_names)}): {class_names}")

    # Train / Test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier (100 Trees, max_depth=16)...")
    start_train = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    t_train = time.time() - start_train
    print(f"[SUCCESS] Model trained in {t_train:.2f} seconds.")

    # Model Evaluation: Training and Test Accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS:")
    print("=" * 65)
    print(f"  - Number of Classes:          {len(class_names)}")
    print(f"  - Total Dataset Samples:      {len(df)}")
    print(f"  - Training Samples:           {len(X_train)}")
    print(f"  - Test Samples:               {len(X_test)}")
    print(f"  - Training Accuracy:          {train_accuracy * 100:.2f}%")
    print(f"  - Validation/Test Accuracy:   {test_accuracy * 100:.2f}%")

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
    header = f"{'':<12}" + "".join([f"{name[:5]:>7}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:11]:<12}" + "".join([f"{val:>7d}" for val in row])
        print(row_str)

    # Top 8 Most Important Features
    if hasattr(model, "feature_importances_"):
        print("\n" + "=" * 65)
        print("TOP INFLUENTIAL FEATURES:")
        print("=" * 65)
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:8]
        for rank, idx in enumerate(top_indices, 1):
            fname = feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
            print(f"  {rank}. {fname:<25} : {importances[idx]:.4f}")

    # Export Model & Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PKL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(config.MODEL_PKL_PATH.name),
        "model_type": "RandomForestClassifier",
        "num_classes": len(class_names),
        "num_features": X.shape[1],
        "classes": class_names,
        "kannada_translations": {c: config.GESTURES_KANNADA.get(c, "") for c in class_names},
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "confusion_matrix": cm.tolist()
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("EXPORT COMPLETED:")
    print(f"  - Model File:         {config.MODEL_PKL_PATH}")
    print(f"  - Label Encoder:      {config.LABEL_ENCODER_PATH}")
    print(f"  - Training Metadata:  {config.MODEL_METADATA_PATH}")
    print("=" * 65)

    return {
        "success": True,
        "model": model,
        "label_encoder": label_encoder,
        "classes": class_names,
        "num_classes": len(class_names),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy": float(cv_scores.mean()),
        "confusion_matrix": cm,
        "report": report_text
    }


if __name__ == "__main__":
    train_gesture_classifier()
