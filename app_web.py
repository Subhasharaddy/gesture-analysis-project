import os
import time
import collections
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import config
from utils.gesture_engine import GestureEngine
from utils.landmark_processor import LandmarkProcessor
from utils.tts_engine import TextToSpeechWorker
from utils.gesture_database import get_gesture_info

DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")
app = Flask(__name__, static_folder=DIST_DIR if os.path.exists(DIST_DIR) else None)
# Allow CORS so the Vite frontend (React) running on a different port can access the API
CORS(app)

# Initialize Unified AI Gesture Engine
print("Initializing Gesture Engine...")
gesture_engine = GestureEngine(
    model_path=config.MODEL_PKL_PATH,
    label_encoder_path=config.LABEL_ENCODER_PATH,
    stability_window=config.STABILITY_WINDOW_SIZE,
    high_threshold=config.CONFIDENCE_HIGH_THRESHOLD,
    medium_threshold=config.CONFIDENCE_MEDIUM_THRESHOLD
)

# Initialize Dedicated Text-To-Speech Engine for Web Backend
print("Initializing Text-To-Speech Engine...")
tts = TextToSpeechWorker(
    speech_map=config.GESTURE_SPEECH_MAP,
    cooldown_seconds=config.SPEECH_COOLDOWN_SECONDS
)

from types import SimpleNamespace

@app.route("/api/predict", methods=["POST"])
def predict_gesture():
    try:
        data = request.json or {}
        hands_input = data.get("hands", [])
        detected_hands = []

        for hand_data in hands_input:
            label = hand_data.get("label", "Right")
            lms = hand_data.get("landmarks", [])
            if len(lms) == 21:
                lms_list = [SimpleNamespace(x=float(lm['x']), y=float(lm['y']), z=float(lm.get('z', 0.0))) for lm in lms]
                detected_hands.append(SimpleNamespace(
                    landmark=lms_list,
                    handedness=label
                ))

        image_shape = data.get("image_shape", [480, 640])
        res = gesture_engine.process_frame(detected_hands, image_shape=tuple(image_shape))

        stable_gesture = res.get("stable_gesture") or res.get("gesture_name")
        raw_pred = res.get("raw_prediction")
        raw_conf = res.get("raw_confidence", 0.0)
        conf = res.get("confidence", 0.0)
        class_id = res.get("class_id", -1)

        candidate = stable_gesture
        if candidate in (None, "", "Detecting...", "Unknown Gesture", "No hand detected", "STANDBY"):
            if raw_pred and raw_pred not in ("Detecting...", "Unknown Gesture", "No hand detected", "STANDBY") and raw_conf >= 0.25:
                candidate = raw_pred
                conf = raw_conf
        if not candidate:
            candidate = "No hand detected" if len(detected_hands) == 0 else "Detecting..."

        print(
            f"[BACKEND_PREDICTION] class_id={class_id} | confidence={conf:.2f} | "
            f"mapped='{raw_pred}' | stabilized='{candidate}'"
        )

        is_emg = res.get("is_emergency", False)
        if candidate not in ("No hand detected", "Detecting...", "Unknown Gesture", "STANDBY", "UNKNOWN GESTURE"):
            tts.speak(candidate, is_emergency=is_emg)

        info = get_gesture_info(candidate)
        
        return jsonify({
            "gesture": candidate,
            "gesture_name": candidate,
            "stable_gesture": candidate,
            "class_id": int(class_id) if hasattr(class_id, "item") else class_id,
            "raw_prediction": raw_pred,
            "confidence": float(conf),
            "confidence_tier": res.get("confidence_tier", "HIGH CONFIDENCE" if conf >= 0.8 else "MEDIUM CONFIDENCE"),
            "is_new_confirmation": res.get("is_new_confirmation", False),
            "meaning": res.get("meaning") or info.get("meaning", ""),
            "kannada": res.get("kannada") or info.get("kannada", ""),
            "spoken_phrase": res.get("spoken_phrase") or info.get("speech", candidate),
            "is_emergency": is_emg or info.get("is_emergency", False)
        })
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": err_str}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    err_str = traceback.format_exc()
    return jsonify({"error": str(e), "traceback": err_str}), 500

@app.route("/api/gestures", methods=["GET"])
def get_gestures():
    """Returns all 42 gestures for the frontend."""
    from utils.gesture_database import GESTURE_DEFINITIONS
    return jsonify(GESTURE_DEFINITIONS)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    if os.path.exists(DIST_DIR):
        file_path = os.path.join(DIST_DIR, path)
        if path and os.path.exists(file_path):
            return send_from_directory(DIST_DIR, path)
        return send_from_directory(DIST_DIR, "index.html")
    return redirect("http://localhost:5173/")

if __name__ == "__main__":
    print("Starting Web Backend API on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)
