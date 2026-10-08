"""
Web Exhibition Dashboard Server for AI Sign Language Recognition (12 Classes).
100% self-contained on Windows laptop:
  - Video Feed: Laptop Webcam with Android Phone USB fallback
  - AI Engine: Unified Landmark Feature Extraction, Random Forest Classifier & Dynamic Motion Tracker
  - 12 Gesture Classes: HOME, NAMASTE, HELLO, THANK YOU, YES, NO, HELP, STOP, WATER, FOOD, PLEASE, GOOD
  - Output: English, Unicode Kannada, and Spoken Audio (pyttsx3)
  - Arduino Hardware: Serial LCD display & Buzzer/LED actuation
  - Interactive Gesture Guide: Visual cards & ISL instructions
"""

import cv2
import numpy as np
import collections
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

import config
from utils.hand_tracker import UniversalHandTracker
from utils.tts_engine import TextToSpeechWorker
from utils.camera_manager import CameraManager
from utils.gesture_engine import GestureEngine
from utils.serial_communicator import ArduinoSerialBridge
from utils.gesture_database import GESTURE_DEFINITIONS, CORE_12_GESTURES, get_gesture_info
from train_model import train_sign_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DashboardServer")

app = Flask(__name__, static_folder="static", template_folder="templates")


class SystemState:
    """Thread-safe operational telemetry state."""
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_active = True
        self.hand_detected = False
        self.hand_count = 0
        self.gesture_name = "STANDBY"
        self.confidence = 0.0
        self.confidence_tier = "UNKNOWN"
        self.meaning = "Awaiting hand in camera view"
        self.kannada = "ಸನ್ನೆಗಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ"
        self.speech_text = ""
        self.category = "Sign Language (ISL)"
        self.motion_type = "STATIC"
        self.is_emergency = False
        self.fps = 0.0
        self.is_muted = False
        self.camera_label = "INITIALIZING"
        self.phone_status = "PHONE NOT DETECTED"
        self.phone_connected = False
        self.arduino_status = "INITIALIZING"
        self.is_training = False

        self.history = collections.deque(maxlen=20)
        self.last_confirmed_gesture = None
        self.last_confirmed_time = 0.0


state = SystemState()

# Initialize Unified AI Gesture Engine
logger.info("Initializing Unified AI Gesture Engine for Web Exhibition...")
gesture_engine = GestureEngine(
    model_path=config.MODEL_PKL_PATH,
    label_encoder_path=config.LABEL_ENCODER_PATH,
    stability_window=config.STABILITY_WINDOW_SIZE,
    high_threshold=config.CONFIDENCE_HIGH_THRESHOLD,
    medium_threshold=config.CONFIDENCE_MEDIUM_THRESHOLD
)

# Threaded Text-to-Speech (Laptop speakers)
tts = TextToSpeechWorker(
    speech_map=config.GESTURE_SPEECH_MAP,
    cooldown_seconds=config.SPEECH_COOLDOWN_SECONDS
)

# Arduino Serial Bridge
arduino = ArduinoSerialBridge(
    port=None,
    baud_rate=config.ARDUINO_BAUD_RATE,
    auto_detect=config.ARDUINO_AUTO_DETECT,
    enable_mock=config.ENABLE_MOCK_ARDUINO,
    cooldown_seconds=config.ARDUINO_COOLDOWN_SECONDS
)
state.arduino_status = arduino.get_status_text()

# Hand Tracker (Supports up to 2 hands)
hand_tracker = UniversalHandTracker(
    max_num_hands=config.MP_MAX_NUM_HANDS,
    min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
)

# Smart Camera Manager
cam_mgr = CameraManager(
    preference=config.CAMERA_PREFERENCE,
    laptop_index=config.LAPTOP_CAMERA_INDEX,
    phone_index=config.PHONE_CAMERA_INDEX,
    phone_stream_url=config.PHONE_STREAM_URL,
    target_width=config.CAMERA_WIDTH,
    target_height=config.CAMERA_HEIGHT
)
state.camera_label = cam_mgr.source_label
state.phone_status = cam_mgr.get_phone_status_text()
state.phone_connected = cam_mgr.is_phone_connected()

latest_jpeg_frame = None
camera_running = True


def camera_loop():
    """Background thread processing video frames and running AI inference."""
    global latest_jpeg_frame, camera_running
    logger.info(f"Starting video processing thread with {cam_mgr.source_label}...")

    frame_cnt = 0
    t_start = time.time()
    current_fps = 30.0

    while camera_running:
        if not state.camera_active:
            time.sleep(0.05)
            continue

        ret, frame = cam_mgr.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)  # Natural selfie orientation
        h, w = frame.shape[:2]

        frame_cnt += 1
        if frame_cnt >= 15:
            now_t = time.time()
            current_fps = frame_cnt / max(0.001, (now_t - t_start))
            with state.lock:
                state.fps = current_fps
                state.camera_label = cam_mgr.source_label
                state.phone_status = cam_mgr.get_phone_status_text()
                state.phone_connected = cam_mgr.is_phone_connected()
                state.arduino_status = arduino.get_status_text()
            frame_cnt = 0
            t_start = now_t

        # 1. Hand Tracking
        detected_hands = hand_tracker.process(frame)

        # Draw skeletal lines & joints
        for hand_obj in detected_hands:
            UniversalHandTracker.draw_landmarks(frame, hand_obj)

        # 2. Unified AI Gesture Engine Evaluation (Supports 12 classes & bimanual signs)
        res = gesture_engine.process_frame(detected_hands, image_shape=(h, w))

        with state.lock:
            state.hand_detected = len(detected_hands) > 0
            state.hand_count = len(detected_hands)
            state.gesture_name = res["confirmed_gesture"]
            state.confidence = res["confidence"]
            state.confidence_tier = res["confidence_tier"]
            state.meaning = res["meaning"]
            state.kannada = res["kannada"]
            state.speech_text = res["spoken_phrase"]
            state.category = res["category"]
            state.motion_type = res["motion_type"]
            state.is_emergency = res["is_emergency"]

            # On new stable gesture confirmation
            if res["is_new_confirmation"] and res["confirmed_gesture"] not in ("UNKNOWN GESTURE", "Gesture not recognized", "STANDBY"):
                timestamp_str = datetime.now().strftime("%H:%M")
                conf_pct = int(res["confidence"] * 100)
                state.history.appendleft({
                    "sign": res["confirmed_gesture"],
                    "time": timestamp_str,
                    "confidence": f"{conf_pct}%",
                    "kannada": res["kannada"],
                    "meaning": res["meaning"]
                })

                if not state.is_muted:
                    tts.speak(res["confirmed_gesture"], is_emergency=res["is_emergency"])

                # Send to Arduino Hardware
                arduino.send_gesture(res["confirmed_gesture"], is_emergency=res["is_emergency"])

        # 3. Visualization Overlays on frame
        for hand_info in res["hands_info"]:
            bx1, by1, bx2, by2 = hand_info["bbox"]
            label = hand_info["label"]
            box_col = (0, 210, 255) if label == "Right" else (255, 120, 0)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_col, 2)
            cv2.putText(frame, f"HAND: {label.upper()}", (bx1, max(14, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA)

        # Top HUD Banner
        hud_bg_color = (20, 24, 33)
        cv2.rectangle(frame, (0, 0), (w, 36), hud_bg_color, -1)
        hud_text = f"FPS: {current_fps:.1f} | HANDS: {len(detected_hands)} | AI: {res['confirmed_gesture']} ({int(res['confidence']*100)}%)"
        tier_color = (46, 160, 67) if res["confidence_tier"] == "HIGH" else ((240, 136, 62) if res["confidence_tier"] == "MEDIUM" else (139, 148, 158))
        if res["is_emergency"]:
            tier_color = (40, 40, 230)
        cv2.putText(frame, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (w - 18, 18), 7, tier_color, -1)

        # Compress to JPEG
        ret_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc:
            latest_jpeg_frame = buffer.tobytes()

        time.sleep(0.005)


video_thread = threading.Thread(target=camera_loop, daemon=True)
video_thread.start()


# --- FLASK ROUTES ---

@app.route("/")
def index():
    """Renders main exhibition dashboard page."""
    return render_template("index.html")


@app.route("/static/images/gestures/<path:filename>")
def serve_gesture_image(filename):
    """Serves gesture guide card images."""
    return send_from_directory(config.IMAGES_DIR, filename)


def mjpeg_generator():
    """Streams MJPEG frames to browser."""
    global latest_jpeg_frame
    while True:
        if latest_jpeg_frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg_frame + b"\r\n")
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def get_status():
    """JSON API returning real-time continuous recognition telemetry, Kannada, and history."""
    with state.lock:
        return jsonify({
            "camera_active": state.camera_active,
            "hand_detected": state.hand_detected,
            "hand_count": state.hand_count,
            "gesture": state.gesture_name,
            "confidence": state.confidence,
            "confidence_tier": state.confidence_tier,
            "meaning": state.meaning,
            "kannada": state.kannada,
            "speech_text": state.speech_text,
            "category": state.category,
            "motion_type": state.motion_type,
            "is_emergency": state.is_emergency,
            "fps": state.fps,
            "is_muted": state.is_muted,
            "camera_label": state.camera_label,
            "phone_status": state.phone_status,
            "phone_connected": state.phone_connected,
            "arduino_status": state.arduino_status,
            "is_training": state.is_training,
            "history": list(state.history)
        })


@app.route("/api/gesture_guide")
def get_gesture_guide():
    """Returns guide data for all 12 classes with images and ISL instructions."""
    guide_items = []
    for g in CORE_12_GESTURES:
        info = get_gesture_info(g)
        safe_name = g.replace(" ", "_").replace("/", "_")
        guide_items.append({
            "name": g,
            "english": info.get("english", g),
            "kannada": info.get("kannada", ""),
            "kannada_translit": info.get("kannada_translit", ""),
            "category": info.get("category", "Sign Language (ISL)"),
            "is_emergency": info.get("is_emergency", False),
            "how_to": info.get("how_to_perform", info.get("description", "")),
            "image_url": f"/static/images/gestures/{safe_name}.png"
        })
    return jsonify({"gestures": guide_items})


@app.route("/api/train_model", methods=["POST"])
def trigger_training():
    """Triggers background ML model training."""
    if state.is_training:
        return jsonify({"success": False, "message": "Training already in progress."})

    def run_train():
        with state.lock:
            state.is_training = True
        try:
            res = train_sign_classifier()
            if res.get("success"):
                gesture_engine.set_model(res["model"], res["label_encoder"])
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            with state.lock:
                state.is_training = False

    t = threading.Thread(target=run_train, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Model training started in background thread."})


@app.route("/api/start_camera", methods=["POST"])
def start_camera_endpoint():
    with state.lock:
        state.camera_active = True
    return jsonify({"success": True, "camera_active": True})


@app.route("/api/stop_camera", methods=["POST"])
def stop_camera_endpoint():
    with state.lock:
        state.camera_active = False
    return jsonify({"success": True, "camera_active": False})


@app.route("/api/switch_camera", methods=["POST"])
def switch_camera():
    """Switches active camera source between Laptop Webcam and Phone USB Camera."""
    success, new_label = cam_mgr.switch_source()
    with state.lock:
        state.camera_label = new_label
        state.phone_status = cam_mgr.get_phone_status_text()
        state.phone_connected = cam_mgr.is_phone_connected()
    return jsonify({
        "success": success,
        "camera_label": new_label,
        "phone_status": state.phone_status,
        "phone_connected": state.phone_connected
    })


@app.route("/api/toggle_mute", methods=["POST"])
def toggle_mute():
    """Mutes / unmutes laptop speech audio."""
    with state.lock:
        state.is_muted = not state.is_muted
        status_text = "MUTED" if state.is_muted else "ACTIVE"
    logger.info(f"Laptop audio toggled to: {status_text}")
    return jsonify({"success": True, "is_muted": state.is_muted})


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Clears gesture history queue."""
    with state.lock:
        state.history.clear()
        state.last_confirmed_gesture = None
    return jsonify({"success": True})


if __name__ == "__main__":
    port = 5000
    print("\n" + "=" * 70)
    print("AI SIGN LANGUAGE RECOGNITION (12 CLASSES) - EXHIBITION WEB SERVER")
    print("=" * 70)
    print("Continuous AI Recognition • Hand Landmarks • Kannada + Audio • Arduino")
    print(f"Open browser: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
