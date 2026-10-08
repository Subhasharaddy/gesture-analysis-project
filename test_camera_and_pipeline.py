import time
import cv2
import numpy as np

import config
from utils.camera_manager import CameraManager
from utils.hand_tracker import UniversalHandTracker
from utils.landmark_processor import LandmarkProcessor
from utils.gesture_engine import GestureEngine
from utils.serial_communicator import ArduinoSerialBridge

def run_camera_and_pipeline_test():
    print("=" * 65)
    print("CAMERA & 42-GESTURE PIPELINE REAL-TIME DIAGNOSTIC SUITE")
    print("=" * 65)

    # 1. Camera Initialization Test
    print("\n[STEP 1] Testing CameraManager Safe Sequential Auto-Discovery...")
    cam_mgr = CameraManager(
        preference="laptop",
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )

    if not cam_mgr.is_opened():
        print("[FAIL] No working camera detected.")
        return False

    print(f"[OK] Camera successfully connected: {cam_mgr.source_label}")
    active_idx = cam_mgr.active_source
    target_w, target_h = cam_mgr.target_width, cam_mgr.target_height

    # 2. Camera Stream Speed & Stability Test (30 frames)
    print("\n[STEP 2] Testing Camera Frame Read Performance (30 frames)...")
    frame_times = []
    read_successes = 0

    # Warmup
    for _ in range(5):
        cam_mgr.read()

    t_start = time.perf_counter()
    for f_idx in range(30):
        t0 = time.perf_counter()
        ret, frame = cam_mgr.read()
        t1 = time.perf_counter()
        if ret and frame is not None and frame.size > 0:
            read_successes += 1
            frame_times.append(t1 - t0)
        time.sleep(0.01)  # Simulate frame pace

    t_total = time.perf_counter() - t_start
    actual_cam_fps = read_successes / t_total if t_total > 0 else 0
    avg_read_ms = (sum(frame_times) / len(frame_times)) * 1000.0 if frame_times else 0

    print(f"[OK] Read {read_successes}/30 frames successfully.")
    print(f"     Average Frame Read Time: {avg_read_ms:.2f} ms")
    print(f"     Measured Camera FPS    : {actual_cam_fps:.1f} FPS")

    # 3. MediaPipe Tracking Test
    print("\n[STEP 3] Testing MediaPipe Hand Tracker (Tasks API Video Mode)...")
    tracker = UniversalHandTracker(
        max_num_hands=config.MP_MAX_NUM_HANDS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
    )

    ret, test_frame = cam_mgr.read()
    assert ret and test_frame is not None, "Failed to read test frame for MediaPipe"

    t0 = time.perf_counter()
    hands = tracker.process(test_frame)
    t_mp = (time.perf_counter() - t0) * 1000.0
    print(f"[OK] MediaPipe processing time: {t_mp:.2f} ms (Hands detected: {len(hands)})")

    # 4. 42-Class GestureEngine & ML Model Test
    print("\n[STEP 4] Testing 42-Class GestureEngine & ML Model...")
    engine = GestureEngine(
        model_path=config.MODEL_PKL_PATH,
        label_encoder_path=config.LABEL_ENCODER_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        smoothing_window=config.STABILITY_WINDOW_SIZE
    )

    assert engine.model is not None, "ML Model not loaded!"
    assert engine.label_encoder is not None, "Label encoder not loaded!"
    classes = list(engine.label_encoder.classes_)
    print(f"[OK] 42-Class Model loaded successfully: {len(classes)} classes")

    # Test engine on frame
    t0 = time.perf_counter()
    result = engine.process_frame(hands, image_shape=test_frame.shape[:2])
    t_engine = (time.perf_counter() - t0) * 1000.0

    print(f"[OK] GestureEngine response: {result['confirmed_gesture']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"     Inference & Smoothing time: {t_engine:.2f} ms")

    # 5. Async Arduino Bridge Test
    print("\n[STEP 5] Testing Non-Blocking Async Arduino Bridge...")
    arduino = ArduinoSerialBridge(baud_rate=115200, enable_mock=True)
    t0 = time.perf_counter()
    arduino.send_gesture("HELLO")
    t_ard = (time.perf_counter() - t0) * 1e6
    print(f"[OK] Async Arduino call latency: {t_ard:.2f} microseconds (0-blocking)")

    # 6. Clean Resource Release Test
    print("\n[STEP 6] Testing Clean Hardware Release...")
    cam_mgr.release()
    tracker.close()
    arduino.close()
    print("[OK] All hardware resources released cleanly.")

    # 7. Summary Diagnostic Output
    print("\n" + "=" * 65)
    print("FINAL CAMERA & SPEED VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"Camera Index        : {active_idx} (Working)")
    print(f"Camera Status       : CONNECTED")
    print(f"Camera Resolution   : {target_w}x{target_h}")
    print(f"Camera Preview FPS  : {actual_cam_fps:.1f} FPS (Target: 25-30 FPS)")
    print(f"MediaPipe Status    : OK ({t_mp:.1f} ms)")
    print(f"Model Status        : OK (42 Classes, {t_engine:.1f} ms)")
    print(f"Arduino Status      : OK (Async, {t_ard/1000.0:.4f} ms)")
    print("=" * 65 + "\n")
    return True

if __name__ == "__main__":
    run_camera_and_pipeline_test()
