"""
Configuration module for AI Sign Language Recognition System (Laptop Edition).
Defines core gestures, Kannada translations, confidence thresholds,
challenge parameters, camera indexes, and file paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

IMAGES_DIR = BASE_DIR / "images" / "gestures"

# Ensure directories exist
DATASET_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifact Paths
MODEL_PKL_PATH = MODELS_DIR / "sign_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "sign_label_encoder.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "sign_model_metadata.json"

from utils.gesture_database import GESTURE_DEFINITIONS, build_speech_map, get_all_gesture_names, CORE_12_GESTURES

# Primary 42 Gesture Classes for Sign Language Recognition (12 Existing + 30 Recovered)
CORE_GESTURES_12 = CORE_12_GESTURES
ALL_GESTURES = get_all_gesture_names()
GESTURES = ALL_GESTURES

# Kannada Translations
GESTURES_KANNADA = {g: d["kannada"] for g, d in GESTURE_DEFINITIONS.items()}
GESTURES_KANNADA["UNKNOWN"] = "ಗುರುತಿಸಲಾಗದ ಸಂಕೇತ"

# Display Names (English – Kannada)
GESTURE_DISPLAY_NAMES = {g: f"{g} – {d['kannada']}" for g, d in GESTURE_DEFINITIONS.items()}
GESTURE_DISPLAY_NAMES["UNKNOWN"] = "UNKNOWN – ಗುರುತಿಸಲಾಗಿಲ್ಲ"

# Semantic Descriptions / Meanings
GESTURE_DESCRIPTIONS = {g: d["meaning"] for g, d in GESTURE_DEFINITIONS.items()}
GESTURE_DESCRIPTIONS["UNKNOWN"] = "Hand posture not recognized or low confidence"

# Spoken Phrases for Text-to-Speech (Laptop speakers)
GESTURE_SPEECH_MAP = build_speech_map()

# Emergency / Priority Gestures (triggers buzzer and red alert strobe)
EMERGENCY_GESTURES = {"HELP", "STOP", "EMERGENCY", "POLICE", "CALL FOR HELP"}

# Fallback label when prediction confidence is below threshold
UNKNOWN_GESTURE_LABEL = "Detecting..."
NO_HAND_LABEL = "No hand detected"

# Fast Camera Settings (Optimized for Real-Time Speed & Low Latency)
CAMERA_PREFERENCE = "laptop"     # "laptop", "auto", or "phone"
LAPTOP_CAMERA_INDEX = 0          # Default built-in laptop webcam
PHONE_CAMERA_INDEX = 1           # Android phone connected via USB (UVC / DroidCam / Iriun)
PHONE_STREAM_URL = "http://127.0.0.1:4747/video"  # Direct ADB / DroidCam stream
CAMERA_WIDTH = 640               # Fast 640x480 resolution
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = LAPTOP_CAMERA_INDEX

# Frame Skipping Pipeline Optimization
PROCESS_EVERY_N_FRAMES = 2       # Run AI inference every N frames, preview runs smooth 30 FPS

# MediaPipe Hand Landmark Settings (Optimized for One-Hand Fast Tracking)
MP_MAX_NUM_HANDS = 1             # Fast single-hand recognition
MP_MIN_DETECTION_CONFIDENCE = 0.50
MP_MIN_TRACKING_CONFIDENCE = 0.50

# Recognition Stability & Anti-Flicker Settings
PREDICTION_HISTORY_SIZE = 5        # Short rolling prediction history for majority voting
GESTURE_CONFIRMATION_FRAMES = 3    # Required consistent predictions before switching gesture
NO_HAND_TIMEOUT = 0.8              # Seconds to hold last valid result if hand is temporarily occluded
CONFIDENCE_THRESHOLD = 0.70        # Filter threshold for accepting new candidate gestures
CONFIDENCE_HIGH_THRESHOLD = 0.80   # >= 80% HIGH CONFIDENCE
CONFIDENCE_MEDIUM_THRESHOLD = 0.70 # 70% - 79% MEDIUM CONFIDENCE
PREDICTION_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
STABILITY_WINDOW_SIZE = PREDICTION_HISTORY_SIZE
CHALLENGE_TIME_SECONDS = 10        # Duration for gesture test challenge
SPEECH_COOLDOWN_SECONDS = 2.0      # Minimum seconds between repeated audio outputs

# Arduino Serial Communication Settings (Non-Blocking Async at 115200 Baud)
ARDUINO_BAUD_RATE = 115200
ARDUINO_AUTO_DETECT = True
ENABLE_MOCK_ARDUINO = True
ARDUINO_COOLDOWN_SECONDS = 0.5

# Model Compatibility Aliases
MODEL_PATH = MODEL_PKL_PATH
DATASET_CSV_PATH = DATASET_DIR / "gestures_dataset.csv"

# Dataset Settings
SAMPLES_PER_GESTURE = 350

# Ensure subdirectories for each gesture exist inside dataset/
for g in ALL_GESTURES:
    safe_folder_name = g.replace(" ", "_").replace("/", "_")
    (DATASET_DIR / safe_folder_name).mkdir(parents=True, exist_ok=True)


