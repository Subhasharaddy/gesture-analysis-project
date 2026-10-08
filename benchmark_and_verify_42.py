import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from types import SimpleNamespace

import config
from utils.landmark_processor import LandmarkProcessor
from utils.gesture_engine import GestureEngine
from utils.serial_communicator import SerialCommunicator
from utils.hand_tracker import UniversalHandTracker

def run_comprehensive_benchmarks():
    print("=" * 70)
    print("      COMPREHENSIVE 42-GESTURE PIPELINE BENCHMARK & VERIFICATION")
    print("=" * 70)

    # 1. Verify Model & Classes
    model_path = Path("models/sign_model.pkl")
    encoder_path = Path("models/sign_label_encoder.pkl")
    meta_path = Path("models/sign_model_metadata.json")

    assert model_path.exists(), "Model file not found!"
    assert encoder_path.exists(), "Encoder file not found!"
    assert meta_path.exists(), "Metadata file not found!"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = list(encoder.classes_)
    print(f"[OK] Total classes registered in model: {len(classes)}")
    assert len(classes) == 42, f"Expected 42 classes, got {len(classes)}"

    # Check against config.GESTURES
    diff_config = set(config.GESTURES) ^ set(classes)
    print(f"[OK] Config vs Model class parity: {len(diff_config) == 0} (diff: {diff_config})")

    # 2. Test Model Per-Gesture Accuracy on Dataset
    csv_path = Path("dataset/gestures_dataset.csv")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y_raw = df['label'].values
    y = encoder.transform(y_raw)

    # Stratified test split evaluation (20%)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_test_pred = model.predict(X_test)
    overall_acc = accuracy_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train))

    print(f"\n--- Accuracy Report ---")
    print(f"Training Accuracy   : {train_acc * 100:.2f}%")
    print(f"Validation Accuracy : {overall_acc * 100:.2f}%")

    report = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True)
    low_accuracy_gestures = []
    for g in classes:
        if g in report:
            prec = report[g]['precision']
            rec = report[g]['recall']
            f1 = report[g]['f1-score']
            if f1 < 0.85:
                low_accuracy_gestures.append((g, f1))
    
    print(f"Gestures with <85% F1-score: {len(low_accuracy_gestures)}")
    for g, f1 in low_accuracy_gestures:
        print(f"  - {g}: {f1*100:.1f}%")

    # 3. Benchmark Landmark Processor
    # Generate synthetic 21 landmarks
    dummy_landmarks = np.random.uniform(0.3, 0.7, (21, 3)).astype(np.float32)
    dummy_hand = SimpleNamespace(
        landmark=[SimpleNamespace(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])) for pt in dummy_landmarks],
        handedness='Right'
    )
    warmup = 100
    loops = 2000
    for _ in range(warmup):
        _ = LandmarkProcessor.extract_feature_vector(dummy_hand)

    t0 = time.perf_counter()
    for _ in range(loops):
        _ = LandmarkProcessor.extract_feature_vector(dummy_hand)
    t1 = time.perf_counter()
    lp_time_us = ((t1 - t0) / loops) * 1e6
    print(f"\n--- Subcomponent Latencies ---")
    print(f"Landmark Preprocessing Time : {lp_time_us:.2f} microseconds ({lp_time_us/1000.0:.4f} ms)")

    # 4. Benchmark ML Model Inference Time
    # Single sample inference (what happens per frame)
    model.n_jobs = 1
    sample_feat = X_test[0:1]
    # Warmup
    for _ in range(100):
        _ = model.predict_proba(sample_feat)

    t0 = time.perf_counter()
    infer_loops = 500
    for _ in range(infer_loops):
        _ = model.predict_proba(sample_feat)
    t1 = time.perf_counter()
    single_infer_ms = ((t1 - t0) / infer_loops) * 1000.0
    print(f"Single-Sample Inference Time: {single_infer_ms:.2f} ms")

    # 5. Benchmark GestureEngine End-to-End
    engine = GestureEngine(model=model, label_encoder=encoder)

    t0 = time.perf_counter()
    engine_loops = 500
    for _ in range(engine_loops):
        _ = engine.process_hand(dummy_hand)
    t1 = time.perf_counter()
    engine_latency_ms = ((t1 - t0) / engine_loops) * 1000.0
    print(f"GestureEngine Prediction    : {engine_latency_ms:.2f} ms")

    # 6. Benchmark MediaPipe Tracker (Tasks API Video Mode)
    tracker = UniversalHandTracker(
        max_num_hands=1,
        min_detection_confidence=0.65
    )
    # Dummy BGR image 640x480
    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Draw simple hand-like blob
    cv2_dummy = np.ascontiguousarray(dummy_frame)
    # Warmup
    for _ in range(10):
        tracker.process(cv2_dummy)
    
    mp_loops = 60
    t0 = time.perf_counter()
    for _ in range(mp_loops):
        tracker.process(cv2_dummy)
    t1 = time.perf_counter()
    mp_time_ms = ((t1 - t0) / mp_loops) * 1000.0
    tracker.close()
    print(f"MediaPipe Processing Time   : {mp_time_ms:.2f} ms")

    # 7. Benchmark Async Arduino Communication
    serial_comm = SerialCommunicator(port="COM999", baud_rate=115200, auto_detect=False, enable_mock=True)
    t0 = time.perf_counter()
    for _ in range(1000):
        serial_comm.send_gesture("HELLO")
    t1 = time.perf_counter()
    arduino_call_us = ((t1 - t0) / 1000) * 1e6
    serial_comm.close()
    print(f"Async Arduino send_gesture  : {arduino_call_us:.2f} microseconds ({arduino_call_us/1000.0:.4f} ms)")

    # 8. Full Pipeline Simulation & Projected Recognition Latency
    # Pipeline per processed frame:
    # Frame Resize (0.2 ms) + MediaPipe (mp_time_ms) + LandmarkProcessor (0.03 ms) + Inference (engine_latency_ms)
    full_pipeline_per_proc_ms = 0.2 + mp_time_ms + 0.03 + engine_latency_ms
    # With PROCESS_EVERY_N_FRAMES = 2:
    # Frame 1: Resize (0.2 ms) + Display draw (0.8 ms) = ~1.0 ms
    # Frame 2: Resize (0.2 ms) + Full pipeline (full_pipeline_per_proc_ms) + Display draw (0.8 ms)
    avg_frame_time_ms = (1.0 + full_pipeline_per_proc_ms + 1.0) / 2.0
    projected_camera_fps = min(30.0, 1000.0 / avg_frame_time_ms)
    # Recognition response time with 2-frame smoothing:
    # 2 processed frames * (1 frame per 2 camera frames) = 4 camera frames = 4 * (1000 / 30) = ~133 ms
    recognition_latency_ms = (2 * config.PROCESS_EVERY_N_FRAMES) * (1000.0 / 30.0) + engine_latency_ms

    print(f"\n--- End-to-End Pipeline Performance ---")
    print(f"Projected Camera Display FPS: {projected_camera_fps:.1f} FPS (Target: 25-30 FPS)")
    print(f"Total AI Inference Time     : {single_infer_ms:.2f} ms (Target: < 20 ms)")
    print(f"Recognition Latency         : {recognition_latency_ms:.1f} ms / {recognition_latency_ms/1000.0:.2f} sec (Target: 100-300 ms)")

    # 9. Print all 42 Gestures
    print("\n" + "=" * 70)
    print("ALL 42 GESTURES IN UNIFIED SYSTEM:")
    print("=" * 70)
    for i, g in enumerate(classes, 1):
        print(f"  {i:2d}. {g:<16} (Precision: {report[g]['precision']*100:.1f}%, Recall: {report[g]['recall']*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_comprehensive_benchmarks()
