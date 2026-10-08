"""
Dataset Collection Tool for AI Sign Language Recognition (12 Classes).
Allows easy and intuitive recording of 300-500 skeletal landmark samples
for all 12 gesture classes:
1. HOME
2. NAMASTE
3. HELLO
4. THANK YOU
5. YES
6. NO
7. HELP
8. STOP
9. WATER
10. FOOD
11. PLEASE
12. GOOD
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import config
from utils.landmark_processor import LandmarkProcessor
from utils.hand_tracker import UniversalHandTracker
from utils.camera_manager import CameraManager
from utils.gesture_database import GESTURE_DEFINITIONS, CORE_12_GESTURES

# All 42 Gestures Configuration (12 Existing + 30 Recovered)
EXISTING_12_GESTURES = list(CORE_12_GESTURES)
RECOVERED_30_GESTURES = [
    "BAD", "CALL FOR HELP", "COME", "DOCTOR", "EMERGENCY",
    "FRIEND", "GO", "GOOD MORNING", "GOOD NIGHT", "HOSPITAL",
    "I AM FINE", "LOVE", "OK", "PHONE", "POLICE",
    "SCHOOL", "SORRY", "THANKS", "WAIT", "I / ME",
    "YOU", "WELCOME", "BYE", "THUMBS UP", "THUMBS DOWN",
    "PEACE", "POINT", "FIST", "OPEN PALM", "WAVE"
]

GESTURES = [g for g in EXISTING_12_GESTURES if g not in RECOVERED_30_GESTURES] + RECOVERED_30_GESTURES


def get_existing_counts():
    """Counts number of .npy sample files in each gesture folder."""
    counts = {}
    for g in GESTURES:
        safe_name = g.replace(" ", "_").replace("/", "_")
        folder = config.DATASET_DIR / safe_name
        folder.mkdir(parents=True, exist_ok=True)
        npy_files = list(folder.glob("sample_*.npy"))
        counts[g] = len(npy_files)
    return counts


def display_gesture_menu(existing_counts: dict, active_idx: int = 0):
    print("\n" + "=" * 65)
    print("      AI SIGN LANGUAGE RECOGNITION - DATA COLLECTOR (42 CLASSES)")
    print("=" * 65)
    print("Supported Classes:")
    for i, name in enumerate(GESTURES, 1):
        info = GESTURE_DEFINITIONS.get(name, {})
        kannada = info.get("kannada_translit", "")
        count = existing_counts.get(name, 0)
        marker = "==>" if (i - 1) == active_idx else "   "
        status = "[DONE]" if count >= config.SAMPLES_PER_GESTURE else f"[{count}/{config.SAMPLES_PER_GESTURE}]"
        print(f" {marker} {i:2d}. {name:<14} – {kannada:<12} {status}")
    print("=" * 65)
    print("Controls:")
    print("  [SPACE]     : Start / Stop recording samples")
    print("  [N] / [P]   : Next / Previous gesture")
    print("  [1 - 9, 0]  : Quick-jump to class")
    print("  [C]         : Toggle Camera (Laptop Webcam <-> Phone)")
    print("  [Q] / [ESC] : Save and Exit")
    print("=" * 65 + "\n")


def run_dataset_collector():
    existing_counts = get_existing_counts()
    current_gesture_idx = 0
    display_gesture_menu(existing_counts, current_gesture_idx)

    # Initialize Hand Tracker
    hand_tracker = UniversalHandTracker(
        max_num_hands=config.MP_MAX_NUM_HANDS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
    )

    # Initialize Camera Manager (Laptop with Phone USB fallback)
    cam_mgr = CameraManager(
        preference=config.CAMERA_PREFERENCE,
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )

    if not cam_mgr.is_opened():
        print("[ERROR] Could not open camera! Check laptop webcam or connect Android phone via USB.")
        hand_tracker.close()
        return

    collecting = False
    countdown_active = False
    countdown_start_time = 0
    collected_in_batch = 0
    TARGET_BATCH = 400

    last_save_time = 0.0

    while True:
        ret, frame = cam_mgr.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # Selfie mirror
        h, w = frame.shape[:2]

        current_gesture = CORE_12_GESTURES[current_gesture_idx]
        info = GESTURE_DEFINITIONS.get(current_gesture, {})
        current_kannada = info.get("kannada_translit", "")
        how_to = info.get("how_to_perform", info.get("description", "Hold sign steady."))
        now = time.time()

        # Handle Countdown logic
        if countdown_active:
            elapsed = now - countdown_start_time
            remaining = 3.0 - elapsed
            if remaining <= 0:
                countdown_active = False
                collecting = True
                collected_in_batch = 0
                print(f"[RECORDING] Collecting {TARGET_BATCH} samples for '{current_gesture}'...")
            else:
                # Display countdown box
                box_w = 320
                box_h = 160
                cx, cy = w // 2, h // 2
                cv2.rectangle(frame, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2), (20, 24, 33), -1)
                cv2.rectangle(frame, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2), (0, 210, 255), 2)
                count_str = str(int(np.ceil(remaining)))
                cv2.putText(frame, count_str, (cx - 20, cy + 15), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 210, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Get Ready: {current_gesture}", (cx - 110, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Detect hands & landmarks
        detected_hands = hand_tracker.process(frame)
        hand_detected = bool(detected_hands)

        if detected_hands:
            for hand in detected_hands:
                UniversalHandTracker.draw_landmarks(frame, hand)

            # Record samples if actively collecting
            if collecting and (now - last_save_time >= 0.035):  # Capture at ~28 FPS to avoid identical dupes
                primary_hand = detected_hands[0]
                feat_vector = LandmarkProcessor.extract_feature_vector(primary_hand)

                safe_name = current_gesture.replace(" ", "_").replace("/", "_")
                gesture_dir = config.DATASET_DIR / safe_name
                gesture_dir.mkdir(parents=True, exist_ok=True)
                existing_files = list(gesture_dir.glob("sample_*.npy"))
                next_idx = len(existing_files) + 1
                file_path = gesture_dir / f"sample_{next_idx:04d}.npy"
                np.save(str(file_path), feat_vector)

                collected_in_batch += 1
                existing_counts[current_gesture] = len(existing_files) + 1
                last_save_time = now

                if collected_in_batch >= TARGET_BATCH:
                    collecting = False
                    print(f"[SUCCESS] Recorded {collected_in_batch} samples for {current_gesture}! Total: {existing_counts[current_gesture]}")
                    display_gesture_menu(existing_counts, current_gesture_idx)

        # --- Top HUD Header Banner ---
        cv2.rectangle(frame, (0, 0), (w, 75), (18, 22, 28), -1)
        cv2.line(frame, (0, 75), (w, 75), (50, 60, 75), 1)

        cv2.putText(frame, "AI SIGN LANGUAGE DATA COLLECTOR", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        
        status_color = (80, 200, 80) if existing_counts[current_gesture] >= config.SAMPLES_PER_GESTURE else (0, 210, 255)
        is_phone = "PHONE" in cam_mgr.source_label.upper()
        display_cam = "PHONE USB" if is_phone else "LAPTOP WEBCAM"

        header_info = f"Class [{current_gesture_idx + 1}/12]: {current_gesture} ({current_kannada}) | Samples: {existing_counts[current_gesture]}/{TARGET_BATCH} | Cam: {display_cam}"
        cv2.putText(frame, header_info, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 1, cv2.LINE_AA)

        # Instructions Banner
        cv2.putText(frame, f"How to sign: {how_to}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 205), 1, cv2.LINE_AA)

        # Bottom Controls / Progress Bar
        cv2.rectangle(frame, (0, h - 55), (w, h), (18, 22, 28), -1)
        cv2.line(frame, (0, h - 55), (w, h - 55), (50, 60, 75), 1)

        if collecting:
            progress_pct = min(1.0, collected_in_batch / float(TARGET_BATCH))
            bar_w = int((w - 40) * progress_pct)
            cv2.rectangle(frame, (20, h - 42), (w - 20, h - 22), (30, 36, 46), -1)
            cv2.rectangle(frame, (20, h - 42), (20 + bar_w, h - 22), (46, 160, 67), -1)
            cv2.putText(frame, f"RECORDING IN PROGRESS: {collected_in_batch}/{TARGET_BATCH} ({int(progress_pct * 100)}%) – Press [SPACE] to pause",
                        (w // 2 - 200, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            prompt_str = f"[SPACE] Record | [N] Next Sign | [P] Prev Sign | [1-9,0] Jump | [C] Cam | [Q] Exit"
            cv2.putText(frame, prompt_str, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

        cv2.imshow("AI Sign Language Data Collector (12 Classes)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32:  # SPACE: Toggle start/pause
            if collecting:
                collecting = False
                print("[PAUSED] Collection paused.")
            elif not countdown_active:
                countdown_active = True
                countdown_start_time = time.time()
        elif key in (ord('n'), ord('N'), 83):  # 'N' or Right arrow -> Next gesture
            current_gesture_idx = (current_gesture_idx + 1) % len(CORE_12_GESTURES)
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif key in (ord('p'), ord('P'), 81):  # 'P' or Left arrow -> Prev gesture
            current_gesture_idx = (current_gesture_idx - 1) % len(CORE_12_GESTURES)
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif ord('1') <= key <= ord('9'):
            current_gesture_idx = (key - ord('1'))
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif key == ord('0'):
            current_gesture_idx = 9  # 10th gesture (FOOD)
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif key == ord('-'):
            current_gesture_idx = 10  # 11th gesture (PLEASE)
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif key == ord('='):
            current_gesture_idx = 11  # 12th gesture (GOOD)
            collecting = False
            countdown_active = False
            display_gesture_menu(existing_counts, current_gesture_idx)
        elif key in (ord('c'), ord('C')):
            success, label = cam_mgr.switch_source()
            print(f"[CAMERA] Switched: {label}")

    cam_mgr.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    print("\nDataset collection ended. Summary:")
    for g, cnt in get_existing_counts().items():
        print(f"  - {g:<11} : {cnt} samples")


if __name__ == "__main__":
    run_dataset_collector()
