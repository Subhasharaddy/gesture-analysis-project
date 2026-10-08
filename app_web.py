import os
import time
import collections
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import config
from utils.gesture_engine import GestureEngine
from utils.landmark_processor import LandmarkProcessor

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

from types import SimpleNamespace

@app.route("/api/predict", methods=["POST"])
def predict_gesture():
    """
    Expects JSON:
    {
      "hands": [
        {
          "label": "Right",
          "landmarks": [ {"x": 0.5, "y": 0.5, "z": 0.1}, ... ]
        }
      ]
    }
    """
    data = request.json
    if not data or "hands" not in data or len(data["hands"]) == 0:
        return jsonify({
            "gesture": "No hand detected",
            "gesture_name": "No hand detected",
            "stable_gesture": "No hand detected",
            "confidence": 0.0,
            "is_new_confirmation": False,
            "meaning": "",
            "kannada": ""
        })
    
    detected_hands = []
    for hand_data in data["hands"]:
        label = hand_data.get("label", "Right")
        lms = hand_data.get("landmarks", [])
        if len(lms) == 21:
            lms_list = [SimpleNamespace(x=float(lm['x']), y=float(lm['y']), z=float(lm.get('z', 0.0))) for lm in lms]
            detected_hands.append(SimpleNamespace(
                landmark=lms_list,
                handedness=label
            ))
            
    if not detected_hands:
        return jsonify({
            "gesture": "No hand detected",
            "gesture_name": "No hand detected",
            "stable_gesture": "No hand detected",
            "confidence": 0.0,
            "is_new_confirmation": False,
            "meaning": "",
            "kannada": ""
        })


    # Pass the mock hands to the existing gesture engine
    # We pass a dummy image_shape since we are not drawing bounding boxes here
    res = gesture_engine.process_frame(detected_hands, image_shape=(480, 640))
    
    candidate = (
        res.get("stable_gesture")
        or res.get("gesture_name")
        or res.get("confirmed_gesture")
        or res.get("gesture")
        or res.get("raw_prediction")
        or "No hand detected"
    )
        
    return jsonify({
        "gesture": candidate,
        "gesture_name": candidate,
        "stable_gesture": candidate,
        "confidence": res.get("confidence", 0.0),
        "confidence_tier": res.get("confidence_tier", "UNKNOWN"),
        "is_new_confirmation": res.get("is_new_confirmation", False),
        "meaning": res.get("meaning", ""),
        "kannada": res.get("kannada", "")
    })

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
    print("Starting Web Backend API on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
