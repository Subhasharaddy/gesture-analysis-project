"""
Real-Time Exhibition Application: AI Sign Language Recognition & Voice Communicator.
Integrates Webcam, MediaPipe Hands, Machine Learning Inference, Threaded TTS,
Hardware Serial Communication (Arduino LCD/Buzzer/LED), and Exhibition HUD Overlay.
"""

import cv2
import mediapipe as mp
import numpy as np
import collections
import time
import sys
import logging
from pathlib import Path
import joblib

import config
from utils.landmark_processor import LandmarkProcessor
from utils.hand_tracker import UniversalHandTracker
from utils.tts_engine import TextToSpeechWorker
from utils.serial_communicator import ArduinoSerialBridge
from utils.ui_overlay import ExhibitionUIOverlay
from utils.camera_manager import CameraManager
from utils.gesture_engine import GestureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExhibitionApp")


def main():
    print("=" * 70)
    print("AI-BASED SIGN LANGUAGE RECOGNITION & VOICE COMMUNICATION SYSTEM")
    print("=" * 70)
    print("Initializing system components for Engineering Exhibition demonstration...")

    # 1. Verify Model Files
    model_path = getattr(config, "MODEL_PATH", config.MODEL_PKL_PATH)
    label_path = config.LABEL_ENCODER_PATH
    if not model_path.exists() or not label_path.exists():
        print("\n[ERROR] Trained model artifacts not found!")
        print(f"Missing: {model_path} or {label_path}")
        print("\nPlease run the training pipeline first:")
        print("  python train_model.py")
        print("=" * 70)
        return

    print("Loading Machine Learning Model & Label Encoder...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)
    classes = list(label_encoder.classes_)
    print(f"[OK] Model loaded successfully. Supported classes: {classes}")

    # 2. Initialize Text-to-Speech Engine
    print("Initializing Threaded Non-Blocking Text-to-Speech Engine...")
    tts = TextToSpeechWorker(
        speech_map=config.GESTURE_SPEECH_MAP,
        cooldown_seconds=config.SPEECH_COOLDOWN_SECONDS
    )

    # 3. Initialize Arduino Serial Hardware Bridge
    print("Initializing Arduino Serial Bridge...")
    arduino = ArduinoSerialBridge(
        port=None,
        baud_rate=config.ARDUINO_BAUD_RATE,
        auto_detect=config.ARDUINO_AUTO_DETECT,
        enable_mock=config.ENABLE_MOCK_ARDUINO,
        cooldown_seconds=config.ARDUINO_COOLDOWN_SECONDS
    )
    print(f"[OK] {arduino.get_status_text()}")

    # 4. Initialize AI Gesture Engine
    print("Initializing Continuous AI Gesture Engine...")
    gesture_engine = GestureEngine(
        model=model,
        label_encoder=label_encoder,
        smoothing_window=config.STABILITY_WINDOW_SIZE,
        high_threshold=config.CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold=config.CONFIDENCE_MEDIUM_THRESHOLD
    )

    # 5. Initialize UI Overlay
    ui = ExhibitionUIOverlay(
        target_gestures=config.GESTURES,
        emergency_gestures=config.EMERGENCY_GESTURES
    )

    # 6. Initialize Hand Tracker
    print("Initializing Universal Hand Tracker...")
    hand_tracker = UniversalHandTracker(
        max_num_hands=config.MP_MAX_NUM_HANDS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
    )

    # 7. Initialize Camera via Central CameraManager (Laptop Webcam with Phone USB Fallback)
    print("Initializing CameraManager...")
    cam_mgr = CameraManager(
        preference=config.CAMERA_PREFERENCE,
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )

    if not cam_mgr.is_opened():
        print(f"[ERROR] Could not access any camera.")
        print("Please check laptop camera permissions or connect Android phone via USB.")
        tts.stop()
        return

    print(f"[OK] Camera stream opened: {cam_mgr.source_label}")
    print(f"[STATUS] Phone Connection: {cam_mgr.get_phone_status_text()}")

    # Operational State Variables
    recent_predictions = collections.deque(maxlen=config.STABILITY_WINDOW_SIZE)
    confirmed_gesture = None
    confirmed_confidence = 0.0
    confirmed_time = 0.0
    is_muted = False

    # FPS Calculation
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    print("\n" + "=" * 70)
    print("SYSTEM READY FOR DEMONSTRATION! EXHIBITION HUD RUNNING.")
    print("Keyboard shortcuts:")
    print("  [Q] or [ESC] : Exit Application")
    print("  [C]         : Switch Camera (Laptop Webcam <-> Phone)")
    print("  [R]         : Reset Current Sign")
    print("  [M]         : Toggle Voice Audio Mute")
    print("  [S]         : Trigger Manual Emergency Alert Test")
    print("=" * 70 + "\n")

    try:
        while cam_mgr.is_opened():
            ret, frame = cam_mgr.read()
            if not ret:
                logger.warning("Dropped frame from video capture.")
                continue

            # Mirror image for intuitive selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Calculate FPS
            frame_count += 1
            if frame_count >= 15:
                now_t = time.time()
                fps = frame_count / (now_t - fps_start_time)
                frame_count = 0
                fps_start_time = now_t

            detected_hands = hand_tracker.process(frame)

            # Skeletal Landmarks Drawing
            for hand_landmarks in detected_hands:
                UniversalHandTracker.draw_landmarks(frame, hand_landmarks)

            # Continuous AI Gesture Recognition via GestureEngine
            res = gesture_engine.process_frame(detected_hands)
            raw_prediction = res["raw_gesture"]
            raw_confidence = res["confidence"]
            confirmed_gesture = res["confirmed_gesture"] if res["confirmed_gesture"] != "STANDBY" else None
            confirmed_confidence = res["confidence"]

            if res["is_new_confirmation"] and confirmed_gesture and confirmed_gesture not in ("UNKNOWN GESTURE", "Gesture not recognized"):
                if not is_muted:
                    tts.speak(confirmed_gesture, is_emergency=res["is_emergency"])
                arduino.send_gesture(confirmed_gesture, is_emergency=res["is_emergency"])

            # Primary hand bounding box
            hand_bbox = None
            if res["hands_info"]:
                hand_bbox = res["hands_info"][0]["bbox"]

            # --- RENDER EXHIBITION HUD OVERLAYS ---

            # 1. Hand brackets & tag
            if hand_bbox is not None:
                display_label = confirmed_gesture if confirmed_gesture else raw_prediction
                ui.draw_hand_brackets(
                    frame,
                    hand_bbox,
                    display_label,
                    confirmed_confidence if confirmed_gesture else raw_confidence
                )

            # 2. Top Banner (System info, FPS, Arduino Status)
            ui.draw_top_bar(
                frame,
                fps=fps,
                arduino_status=arduino.get_status_text(),
                is_muted=is_muted
            )

            # 3. Sidebar (Supported Gestures with Active Indicator)
            ui.draw_side_gesture_panel(frame, current_gesture=confirmed_gesture)

            # 4. Bottom Dock (Recognized word, speech subtitle, confidence meter)
            spoken_phrase = f"{res.get('meaning', '')} ({res.get('kannada_translit', '')})" if confirmed_gesture else ""
            ui.draw_bottom_result_banner(
                frame,
                gesture_name=confirmed_gesture,
                confidence=confirmed_confidence,
                spoken_phrase=spoken_phrase
            )

            # 5. Emergency Strobe Alert if HELP or STOP active
            if confirmed_gesture in config.EMERGENCY_GESTURES:
                ui.draw_emergency_flasher(frame, confirmed_gesture)

            # Display frame
            cv2.imshow("AI Sign Language & Voice Communication System", frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[INFO] Exit command received. Shutting down...")
                break
            elif key == ord('c'):
                print("\n[SWITCH] Switching camera source...")
                success, label = cam_mgr.switch_source()
                print(f"[SWITCH] Active Camera: {label}")
                print(f"[STATUS] {cam_mgr.get_phone_status_text()}\n")
            elif key == ord('r'):
                confirmed_gesture = None
                recent_predictions.clear()
                print("[RESET] Recognition state cleared.")
            elif key == ord('m'):
                is_muted = not is_muted
                print(f"[AUDIO] Voice output {'MUTED' if is_muted else 'ACTIVE'}.")
            elif key == ord('s'):
                print("[TEST] Triggering manual HELP emergency test...")
                confirmed_gesture = "HELP"
                confirmed_confidence = 0.99
                confirmed_time = time.time()
                if not is_muted:
                    tts.speak("HELP", is_emergency=True, force=True)
                arduino.send_gesture("HELP", is_emergency=True, force=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Clean Resource Deallocation
        print("Releasing hardware and background workers...")
        cam_mgr.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        tts.stop()
        arduino.close()
        print("[SUCCESS] System shutdown complete.")


if __name__ == "__main__":
    main()
