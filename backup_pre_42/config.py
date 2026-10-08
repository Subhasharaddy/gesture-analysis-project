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

# Primary 12 Gesture Classes for Sign Language Recognition
CORE_GESTURES_12 = CORE_12_GESTURES
GESTURES = CORE_GESTURES_12
ALL_GESTURES = get_all_gesture_names()

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
EMERGENCY_GESTURES = {"HELP", "STOP"}

# Fallback label when prediction confidence is below threshold
UNKNOWN_GESTURE_LABEL = "Gesture not recognized"

# Camera Settings (Prioritize Built-In Laptop Webcam)
CAMERA_PREFERENCE = "laptop"     # "laptop", "auto", or "phone"
LAPTOP_CAMERA_INDEX = 0          # Default built-in laptop webcam
PHONE_CAMERA_INDEX = 1           # Android phone connected via USB (UVC / DroidCam / Iriun)
PHONE_STREAM_URL = "http://127.0.0.1:4747/video"  # Direct ADB / DroidCam stream
CAMERA_WIDTH = 1280              # HD resolution
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_INDEX = LAPTOP_CAMERA_INDEX

# MediaPipe Hand Landmark Settings (Supports up to 2 hands)
MP_MAX_NUM_HANDS = 2             # Up to 2 hands (Left & Right)
MP_MIN_DETECTION_CONFIDENCE = 0.65
MP_MIN_TRACKING_CONFIDENCE = 0.60

# Recognition Stability & Multi-Tier Confidence Thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.80   # >= 80% HIGH CONFIDENCE
CONFIDENCE_MEDIUM_THRESHOLD = 0.60 # 60% - 79% MEDIUM CONFIDENCE
CONFIDENCE_THRESHOLD = 0.60        # Below this: "Gesture not recognized"
PREDICTION_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
STABILITY_WINDOW_SIZE = 12         # Rolling majority voting window (10-18 frames)
CHALLENGE_TIME_SECONDS = 10        # Duration for gesture test challenge
SPEECH_COOLDOWN_SECONDS = 2.5      # Minimum seconds between repeated audio outputs

# Arduino Serial Communication Settings
ARDUINO_BAUD_RATE = 9600
ARDUINO_AUTO_DETECT = True
ENABLE_MOCK_ARDUINO = True
ARDUINO_COOLDOWN_SECONDS = 1.5

# Model Compatibility Aliases
MODEL_PATH = MODEL_PKL_PATH
DATASET_CSV_PATH = DATASET_DIR / "gestures_dataset.csv"

# Dataset Settings
SAMPLES_PER_GESTURE = 350

# Ensure subdirectories for each gesture exist inside dataset/
for g in ALL_GESTURES:
    safe_folder_name = g.replace(" ", "_").replace("/", "_")
    (DATASET_DIR / safe_folder_name).mkdir(parents=True, exist_ok=True)


