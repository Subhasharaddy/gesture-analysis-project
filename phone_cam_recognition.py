"""
AI Sign Language Recognition using Android Phone as USB Webcam.
Centralizes camera handling through utils.camera_manager.CameraManager:
  - Supports:
    1. Direct USB UVC Webcam Mode (Android 14+ native, index 1 or 2)
    2. DroidCam / Iriun Virtual DirectShow Camera (index 1 or 2)
    3. Direct USB ADB Stream URL (http://127.0.0.1:4747/video)
    4. Real-time switching between Laptop Webcam and Phone Camera with key [C]
    5. Graceful fallback to laptop webcam if phone is disconnected
"""

import cv2
import numpy as np
import collections
import time
import argparse
import sys
from pathlib import Path
import joblib

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import config
from utils.landmark_processor import LandmarkProcessor
from utils.hand_tracker import UniversalHandTracker
from utils.tts_engine import TextToSpeechWorker
from utils.camera_manager import CameraManager


def parse_camera_source(source_arg: str):
    """Parses camera source: integer index (0, 1) or stream URL string or keyword."""
    s = source_arg.strip()
    if s.lower() in ("phone", "laptop", "auto"):
        return s.lower()
    try:
        return int(s)
    except ValueError:
        return s


def draw_hud(
    frame: np.ndarray,
    hand_detected: bool,
    gesture_name: str,
    confidence: float,
    source_label: str,
    phone_status: str,
    fps: float,
    hand_bbox: tuple = None,
    is_muted: bool = False
):
    h, w = frame.shape[:2]

    # 1. Top Header Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (20, 24, 33), -1)
    cv2.line(overlay, (0, 58), (w, 58), (65, 75, 90), 2)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.putText(
        frame,
        "AI SIGN RECOGNITION (USB PHONE CAMERA)",
        (20, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Subtitle with source & FPS
    cv2.putText(
        frame,
        f"Source: {source_label} | FPS: {fps:.1f}",
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 215, 255),
        1,
        cv2.LINE_AA
    )

    # Phone Status Pill
    is_phone_ok = "CONNECTED" in phone_status
    phone_pill_color = (80, 200, 80) if is_phone_ok else (40, 140, 230)
    cv2.rectangle(frame, (w - 410, 14), (w - 225, 44), (25, 30, 42), -1)
    cv2.rectangle(frame, (w - 410, 14), (w - 225, 44), phone_pill_color, 1)
    cv2.putText(
        frame,
        phone_status,
        (w - 400, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        phone_pill_color,
        1,
        cv2.LINE_AA
    )

    # Hand status pill
    status_color = (80, 200, 80) if hand_detected else (140, 140, 140)
    status_text = "HAND: DETECTED" if hand_detected else "HAND: SEARCHING"
    cv2.rectangle(frame, (w - 215, 14), (w - 15, 44), (25, 30, 42), -1)
    cv2.rectangle(frame, (w - 215, 14), (w - 15, 44), status_color, 1)
    cv2.putText(frame, status_text, (w - 205, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

    # 2. Hand Bounding Box & Target Tag
    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
        is_emg = gesture_name in config.EMERGENCY_GESTURES
        box_color = (40, 40, 230) if is_emg else (0, 215, 255)
        corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)

        # Corner brackets
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 2)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 2)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 2)
        cv2.line(frame, (x1, y2), (x1, y1 + corner_len), box_color, 2)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 2)

        badge_text = f"{gesture_name} ({int(confidence * 100)}%)"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 14, y1), box_color, -1)
        cv2.putText(frame, badge_text, (x1 + 6, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Bottom Result Card
    card_h = 95
    card_y = h - card_h
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, card_y), (w, h), (20, 24, 33), -1)
    cv2.line(overlay2, (0, card_y), (w, card_y), (65, 75, 90), 2)
    cv2.addWeighted(overlay2, 0.90, frame, 0.10, 0, frame)

    is_emg = gesture_name in config.EMERGENCY_GESTURES
    text_color = (40, 40, 230) if is_emg else ((80, 200, 80) if gesture_name not in ("AWAITING GESTURE", config.UNKNOWN_GESTURE_LABEL) else (180, 180, 180))

    kannada_trans = config.GESTURES_KANNADA.get(gesture_name, "")
    display_title = f"{gesture_name} – {kannada_trans}" if kannada_trans else gesture_name
    cv2.putText(frame, display_title, (25, card_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

    phrase = config.GESTURE_SPEECH_MAP.get(gesture_name, "")
    if phrase:
        cv2.putText(frame, f"\"{phrase}\"", (25, card_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)

    # Confidence bar
    bar_x = w - 380
    bar_y = card_y + 24
    bar_w = 170
    bar_h = 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 36, 46), -1)
    fill_w = int(bar_w * min(1.0, max(0.0, confidence)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), text_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 75, 90), 1)
    cv2.putText(frame, f"Confidence: {int(confidence * 100)}%", (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Controls footer
    audio_str = "MUTED" if is_muted else "AUDIO ON"
    cv2.putText(
        frame,
        f"[C] Switch Camera  |  [M] {audio_str}  |  [Q] Exit",
        (bar_x, card_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (160, 160, 160),
        1,
        cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser(description="AI Sign Language Recognition using Phone Camera via USB")
    parser.add_argument(
        "--source",
        default=config.CAMERA_PREFERENCE,
        help="Camera source: 'phone', 'laptop', 'auto', 0, 1, or stream URL (e.g. http://127.0.0.1:4747/video)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AI SIGN LANGUAGE RECOGNITION - PHONE USB WEBCAM INTEGRATION")
    print("=" * 70)

    # 1. Verify Model
    model_path = getattr(config, "MODEL_PATH", config.MODEL_PKL_PATH)
    label_path = config.LABEL_ENCODER_PATH
    if not model_path.exists() or not label_path.exists():
        print(f"[ERROR] Trained model artifacts not found at {model_path}")
        print("Please run: python train_model.py")
        return

    print("Loading Machine Learning Model & Label Encoder...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)
    print(f"[OK] Loaded classifier for {len(label_encoder.classes_)} gestures: {list(label_encoder.classes_)}")

    # 2. Text to Speech
    tts = TextToSpeechWorker(speech_map=config.GESTURE_SPEECH_MAP, cooldown_seconds=config.SPEECH_COOLDOWN_SECONDS)

    # 3. Hand Tracker
    hand_tracker = UniversalHandTracker(
        max_num_hands=config.MP_MAX_NUM_HANDS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
    )

    # 4. Central Camera Manager
    parsed_source = parse_camera_source(args.source)
    print(f"[INFO] Initializing CameraManager with source preference: '{parsed_source}'...")

    cam_mgr = CameraManager(
        preference=parsed_source if isinstance(parsed_source, str) else "custom",
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )

    # If explicit numeric index or URL was requested, apply it
    if isinstance(parsed_source, int) or (isinstance(parsed_source, str) and parsed_source.startswith("http")):
        cam_mgr.set_source(parsed_source)

    # Display Camera Selection Menu
    print("\n" + "=" * 45)
    print("Available Cameras:")
    for dev in cam_mgr.device_info:
        idx = dev["index"]
        label = "Laptop Webcam" if idx == config.LAPTOP_CAMERA_INDEX else "Android Phone Camera"
        print(f"{idx} - {label}")

    phone_status = cam_mgr.get_phone_status_text()
    print("=" * 45)
    print(f"Connection Status: {phone_status}")
    print(f"Active Camera:     {cam_mgr.source_label}")
    print("=" * 45)

    if not cam_mgr.is_opened():
        print("[ERROR] No active camera feed could be established.")
        print("Please check laptop camera permissions or connect Android phone via USB.")
        tts.stop()
        hand_tracker.close()
        return

    print("\nSTREAM RUNNING! Keyboard Controls:")
    print("  [C] : Toggle Camera (Switch between Laptop Webcam and Phone)")
    print("  [M] : Toggle Voice Audio Mute")
    print("  [Q] : Exit Application")
    print("=" * 70 + "\n")

    rolling_preds = collections.deque(maxlen=config.STABILITY_WINDOW_SIZE)
    active_gesture = "AWAITING GESTURE"
    active_conf = 0.0
    last_spoken = None
    is_muted = False

    fps = 0.0
    frame_cnt = 0
    t_prev = time.time()

    try:
        while True:
            ret, frame = cam_mgr.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            frame_cnt += 1
            if frame_cnt >= 15:
                now_t = time.time()
                fps = frame_cnt / (now_t - t_prev)
                frame_cnt = 0
                t_prev = now_t

            detected_hands = hand_tracker.process(frame)

            hand_detected = False
            raw_pred = None
            raw_conf = 0.0
            hand_bbox = None

            if detected_hands:
                hand_detected = True
                for hand_landmarks in detected_hands:
                    UniversalHandTracker.draw_landmarks(frame, hand_landmarks)

                    feat_vector = LandmarkProcessor.extract_feature_vector(hand_landmarks)
                    probs = model.predict_proba([feat_vector])[0]
                    best_idx = np.argmax(probs)
                    raw_conf = float(probs[best_idx])
                    pred_label = label_encoder.classes_[best_idx]

                    conf_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 0.70)
                    if raw_conf >= conf_threshold:
                        raw_pred = pred_label
                    else:
                        raw_pred = config.UNKNOWN_GESTURE_LABEL

                    hand_bbox = LandmarkProcessor.get_bounding_box(hand_landmarks, w, h, margin=20)
                    break

            # Temporal smoothing
            if hand_detected and raw_pred:
                rolling_preds.append(raw_pred)
            else:
                if len(rolling_preds) > 0:
                    rolling_preds.popleft()

            if len(rolling_preds) == config.STABILITY_WINDOW_SIZE:
                counts = collections.Counter(rolling_preds)
                candidate, count = counts.most_common(1)[0]
                if count >= config.STABILITY_WINDOW_SIZE - 2:
                    active_gesture = candidate
                    active_conf = raw_conf

                    if candidate != config.UNKNOWN_GESTURE_LABEL:
                        if candidate != last_spoken:
                            last_spoken = candidate
                            is_emg = candidate in config.EMERGENCY_GESTURES
                            if not is_muted:
                                tts.speak(candidate, is_emergency=is_emg)

            elif not hand_detected and len(rolling_preds) == 0:
                active_gesture = "AWAITING GESTURE"
                active_conf = 0.0

            draw_hud(
                frame=frame,
                hand_detected=hand_detected,
                gesture_name=active_gesture,
                confidence=active_conf,
                source_label=cam_mgr.source_label,
                phone_status=cam_mgr.get_phone_status_text(),
                fps=fps,
                hand_bbox=hand_bbox,
                is_muted=is_muted
            )

            cv2.imshow("Phone USB Webcam - AI Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('m'):
                is_muted = not is_muted
                print(f"[AUDIO] Audio voice {'MUTED' if is_muted else 'ACTIVE'}")
            elif key == ord('c'):
                print("\n[SWITCH] Switching camera source...")
                success, new_label = cam_mgr.switch_source()
                print(f"[SWITCH] Active Camera: {new_label}")
                print(f"[STATUS] {cam_mgr.get_phone_status_text()}\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("Cleaning up resources...")
        cam_mgr.release()
        cv2.destroyAllWindows()
        hand_tracker.close()
        tts.stop()
        print("[SUCCESS] Application closed.")


if __name__ == "__main__":
    main()
