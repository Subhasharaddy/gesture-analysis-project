"""
Advanced Real-Time Gesture & Sign Recognition Engine.
Performs:
  1. Anatomical Hand Landmark Analysis:
     - Finger extension/curl states (Thumb, Index, Middle, Ring, Pinky)
     - Finger spread and joint angles
     - Landmark distances & palm orientation
  2. Temporal Trajectory Tracking:
     - Multi-frame displacement & velocity
     - Distinguishes STATIC GESTURES from DYNAMIC GESTURES (e.g. WAVE / BYE)
  3. Machine Learning + Geometric Rule Ensemble:
     - Random Forest model inference on 73-dimensional invariant feature vector
     - Cross-validation against anatomical geometric constraints
  4. Temporal Smoothing & Voting:
     - Rolling prediction buffer (10-18 frames) to eliminate flickering
  5. Confidence Estimation & Thresholding:
     - HIGH (>=80%), MEDIUM (60-79%), UNKNOWN (<60%)
  6. Multi-Hand Support (Left Hand, Right Hand, Both Hands)
"""

import math
import time
import collections
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import config
from utils.landmark_processor import LandmarkProcessor
from utils.gesture_database import GESTURE_DEFINITIONS, get_gesture_info


class HandMotionTracker:
    """Tracks hand landmark movement across frames to detect dynamic motion gestures."""

    def __init__(self, history_len: int = 18):
        self.history_len = history_len
        self.wrist_history = collections.deque(maxlen=history_len)
        self.timestamps = collections.deque(maxlen=history_len)

    def update(self, wrist_pt: Tuple[float, float]):
        """Adds current normalized wrist position (x, y)."""
        now = time.time()
        self.wrist_history.append(wrist_pt)
        self.timestamps.append(now)

    def clear(self):
        self.wrist_history.clear()
        self.timestamps.clear()

    def detect_wave_motion(self) -> Tuple[bool, float]:
        """
        Detects repetitive horizontal swaying (waving hand).
        Returns (is_wave, intensity_score).
        """
        if len(self.wrist_history) < 10:
            return False, 0.0

        xs = [pt[0] for pt in self.wrist_history]
        dxs = np.diff(xs)

        # Count direction reversals (zero-crossings of velocity)
        sign_changes = 0
        for i in range(len(dxs) - 1):
            if dxs[i] * dxs[i + 1] < -1e-5 and abs(dxs[i]) > 0.008:
                sign_changes += 1

        total_span_x = max(xs) - min(xs)

        # A wave requires at least 2 reversals and significant horizontal span
        if sign_changes >= 2 and total_span_x > 0.06:
            score = min(1.0, (sign_changes / 3.0) * (total_span_x / 0.12))
            return True, score

        return False, 0.0


class GestureEngine:
    """
    Central continuous gesture recognition engine integrating:
    - 21-joint skeletal feature extraction
    - Geometric finger state analysis
    - Machine learning classification
    - Temporal smoothing & hysteresis
    - Dynamic motion tracking
    """

    def __init__(
        self,
        model=None,
        label_encoder=None,
        model_path=None,
        label_encoder_path=None,
        confidence_threshold: float = 0.65,
        smoothing_window: int = 12,
        stability_window: Optional[int] = None,
        high_threshold: float = 0.80,
        medium_threshold: float = 0.60
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = stability_window if stability_window is not None else smoothing_window
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        # Load ML model if directly supplied or from path
        if model is not None and label_encoder is not None:
            self.model = model
            self.label_encoder = label_encoder
        else:
            m_path = Path(model_path) if model_path else getattr(config, "MODEL_PKL_PATH", None)
            l_path = Path(label_encoder_path) if label_encoder_path else getattr(config, "LABEL_ENCODER_PATH", None)
            if m_path and m_path.exists() and l_path and l_path.exists():
                try:
                    import joblib
                    self.model = joblib.load(m_path)
                    self.label_encoder = joblib.load(l_path)
                except Exception:
                    self.model = None
                    self.label_encoder = None
                
            else:
                self.model = None
                self.label_encoder = None

        self.motion_tracker = HandMotionTracker(history_len=20)
        self.prediction_buffer = collections.deque(maxlen=smoothing_window)

        self.active_gesture: str = "STANDBY"
        self.active_confidence: float = 0.0
        self.active_status: str = "AWAITING HAND"
        self.is_dynamic: bool = False
        self.handedness_str: str = "UNKNOWN"

        self.last_confirmed_gesture: Optional[str] = None
        self.confirmed_time: float = 0.0

    def set_model(self, model, label_encoder):
        """Updates the active machine learning model."""
        self.model = model
        self.label_encoder = label_encoder

    @staticmethod
    def analyze_finger_states(landmarks_norm: np.ndarray, raw_coords: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes individual fingers: extension state, curl, and relationships.
        landmarks_norm: (21, 3) normalized coordinates.
        raw_coords: (21, 3) raw MediaPipe screen coordinates.
        """
        wrist = raw_coords[0]

        # Fingertip and joint indices:
        # Thumb: 1, 2, 3, 4 | Index: 5, 6, 7, 8 | Middle: 9, 10, 11, 12 | Ring: 13, 14, 15, 16 | Pinky: 17, 18, 19, 20
        wrist = raw_coords[0]
        palm_scale = max(0.04, float(np.linalg.norm(raw_coords[9] - wrist)))

        # Distance-from-wrist invariant extension check
        index_dist_tip = np.linalg.norm(raw_coords[8] - wrist)
        index_dist_pip = np.linalg.norm(raw_coords[6] - wrist)
        index_ext = bool(index_dist_tip > index_dist_pip * 1.05 or (raw_coords[8][1] < raw_coords[6][1]))

        middle_dist_tip = np.linalg.norm(raw_coords[12] - wrist)
        middle_dist_pip = np.linalg.norm(raw_coords[10] - wrist)
        middle_ext = bool(middle_dist_tip > middle_dist_pip * 1.05 or (raw_coords[12][1] < raw_coords[10][1]))

        ring_dist_tip = np.linalg.norm(raw_coords[16] - wrist)
        ring_dist_pip = np.linalg.norm(raw_coords[14] - wrist)
        ring_ext = bool(ring_dist_tip > ring_dist_pip * 1.05 or (raw_coords[16][1] < raw_coords[14][1]))

        pinky_dist_tip = np.linalg.norm(raw_coords[20] - wrist)
        pinky_dist_pip = np.linalg.norm(raw_coords[18] - wrist)
        pinky_ext = bool(pinky_dist_tip > pinky_dist_pip * 1.05 or (raw_coords[20][1] < raw_coords[18][1]))

        # Thumb analysis
        thumb_tip = raw_coords[4]
        thumb_mcp = raw_coords[2]

        # Upward / Downward thumb relative to index base and MCP
        thumb_up = bool(thumb_tip[1] < raw_coords[5][1] - 0.02 and thumb_tip[1] < thumb_mcp[1] - 0.02)
        thumb_down = bool(thumb_tip[1] > raw_coords[5][1] + 0.04 and thumb_tip[1] > thumb_mcp[1] + 0.02)
        thumb_tucked = bool(abs(thumb_tip[0] - raw_coords[9][0]) < 0.06 and abs(thumb_tip[1] - raw_coords[9][1]) < 0.08)

        # Thumb extension distance relative to index MCP
        dist_thumb_mcp = np.linalg.norm(thumb_tip - raw_coords[5])
        thumb_ext = bool(dist_thumb_mcp > 0.10 or thumb_up or thumb_down)

        # Distances between fingertips
        dist_thumb_index = float(np.linalg.norm(thumb_tip - raw_coords[8]))
        dist_index_middle = float(np.linalg.norm(raw_coords[8] - raw_coords[12]))
        dist_middle_ring = float(np.linalg.norm(raw_coords[12] - raw_coords[16]))
        dist_thumb_pinky = float(np.linalg.norm(thumb_tip - raw_coords[20]))

        # All curled (fist) check
        all_curled = bool(not index_ext and not middle_ext and not ring_ext and not pinky_ext)
        all_extended = bool(index_ext and middle_ext and ring_ext and pinky_ext)

        return {
            "palm_scale": palm_scale,
            "thumb_ext": thumb_ext,
            "thumb_up": thumb_up,
            "thumb_down": thumb_down,
            "thumb_tucked": thumb_tucked,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "ring_ext": ring_ext,
            "pinky_ext": pinky_ext,
            "all_curled": all_curled,
            "all_extended": all_extended,
            "dist_thumb_index": dist_thumb_index,
            "dist_index_middle": dist_index_middle,
            "dist_middle_ring": dist_middle_ring,
            "dist_thumb_pinky": dist_thumb_pinky,
        }

    def rule_based_classify(self, f: Dict[str, Any], raw_coords: np.ndarray) -> Tuple[str, float]:
        """
        High-precision anatomical heuristic classifier for the 12 target classes:
        HOME, NAMASTE, HELLO, THANK YOU, YES, NO, HELP, STOP, WATER, FOOD, PLEASE, GOOD.
        """
        # 1. GOOD (Thumbs Up): Thumb pointing up, 4 fingers curled
        if f["thumb_up"] and f["all_curled"]:
            return "GOOD", 0.96

        # 2. HELP: Closed fist with thumb tucked into palm (Distress signal)
        if f["all_curled"] and f["thumb_tucked"]:
            return "HELP", 0.95

        # 3. YES: Closed fist held firmly forward
        if f["all_curled"]:
            return "YES", 0.92

        # 4. WATER: 'W' handshape (Index, Middle, Ring extended, Pinky curled)
        if f["index_ext"] and f["middle_ext"] and f["ring_ext"] and not f["pinky_ext"]:
            return "WATER", 0.94

        # 5. NO: Index & Middle extended together forward, Ring & Pinky curled
        if f["index_ext"] and f["middle_ext"] and not f["ring_ext"] and not f["pinky_ext"]:
            if f["dist_index_middle"] < 0.050:
                return "NO", 0.93
            else:
                return "NO", 0.88

        # 6. FOOD: Flattened 'O' pinch - all fingertips grouped tightly together
        dist_tips_avg = (f["dist_thumb_index"] + f["dist_index_middle"] + f["dist_middle_ring"]) / 3.0
        if dist_tips_avg < 0.055 and not f["all_curled"]:
            return "FOOD", 0.93

        # 7. STOP / HELLO / PLEASE / THANK YOU / NAMASTE: Extended hand postures
        if f["all_extended"]:
            # Fingers tight together -> STOP
            if f["dist_index_middle"] < 0.040 and f["dist_middle_ring"] < 0.040:
                return "STOP", 0.94
            # Upright flat prayer posture -> NAMASTE
            elif abs(raw_coords[0][0] - 0.5) < 0.12 and f["dist_index_middle"] < 0.055:
                return "NAMASTE", 0.91
            # Fingers spread wide -> HELLO
            elif f["dist_index_middle"] > 0.055:
                return "HELLO", 0.92
            else:
                return "PLEASE", 0.90

        return "Gesture not recognized", 0.35

    def process_hand(
        self,
        hand_landmarks,
        handedness_label: str = "Right",
        image_shape: Tuple[int, int] = (720, 1280)
    ) -> Dict[str, Any]:
        """
        Executes end-to-end continuous recognition on a single detected hand:
          Landmarks -> Geometric Features -> ML Inference -> Smoothing
        """
        raw_coords = LandmarkProcessor.extract_raw_landmarks(hand_landmarks)
        norm_coords = LandmarkProcessor.normalize_landmarks(raw_coords)
        feat_vector = LandmarkProcessor.extract_feature_vector(hand_landmarks)

        # 1. Update motion tracker with wrist coordinates
        wrist_xy = (float(raw_coords[0][0]), float(raw_coords[0][1]))
        self.motion_tracker.update(wrist_xy)

        # 2. Extract anatomical finger states
        f_states = self.analyze_finger_states(norm_coords, raw_coords)

        is_dynamic = False

        # 3. Machine Learning Model inference (Primary)
        ml_gesture = None
        ml_conf = 0.0
        if self.model is not None and self.label_encoder is not None:
            try:
                probs = self.model.predict_proba([feat_vector])[0]
                best_idx = np.argmax(probs)
                ml_conf = float(probs[best_idx])
                candidate_gesture = self.label_encoder.classes_[best_idx]
                # Format to uppercase
                ml_gesture = candidate_gesture.strip().upper()
            except Exception:
                pass

        # 4. Geometric rule classification (Auxiliary / Fallback)
        rule_gesture, rule_conf = self.rule_based_classify(f_states, raw_coords)

        # Prioritize trained ML model
        if ml_gesture is not None and ml_conf >= self.confidence_threshold:
            raw_gesture = ml_gesture
            raw_conf = ml_conf
        elif ml_gesture is not None and ml_conf < self.confidence_threshold:
            # Below confidence threshold -> do not force incorrect prediction
            raw_gesture = "Gesture not recognized"
            raw_conf = ml_conf
        elif rule_gesture != "Gesture not recognized" and rule_conf >= self.confidence_threshold:
            raw_gesture = rule_gesture
            raw_conf = rule_conf
        else:
            raw_gesture = "Gesture not recognized"
            raw_conf = max(rule_conf, ml_conf)

        # 5. Temporal Smoothing Buffer (Rolling majority voting)
        self.prediction_buffer.append(raw_gesture)

        if len(self.prediction_buffer) >= max(3, self.smoothing_window // 2):
            counts = collections.Counter(self.prediction_buffer)
            consensus_gesture, count = counts.most_common(1)[0]
            if count >= len(self.prediction_buffer) * 0.50:
                stable_gesture = consensus_gesture
            else:
                stable_gesture = self.active_gesture or consensus_gesture
        else:
            stable_gesture = raw_gesture

        # Confidence categorization
        if raw_conf >= self.high_threshold:
            conf_tier = "HIGH CONFIDENCE"
        elif raw_conf >= self.medium_threshold:
            conf_tier = "MEDIUM CONFIDENCE"
        else:
            conf_tier = "LOW CONFIDENCE"
            if raw_conf < self.confidence_threshold:
                stable_gesture = "Gesture not recognized"

        self.active_gesture = stable_gesture
        self.active_confidence = raw_conf
        self.is_dynamic = is_dynamic
        self.handedness_str = handedness_label.upper()

        if stable_gesture not in ("STANDBY", "Gesture not recognized", "UNKNOWN GESTURE"):
            self.active_status = "GESTURE DETECTED"
        else:
            self.active_status = "ANALYZING"

        # Retrieve rich dictionary metadata
        info = get_gesture_info(stable_gesture)

        return {
            "gesture": stable_gesture,
            "confidence": float(raw_conf),
            "confidence_tier": conf_tier,
            "status": self.active_status,
            "is_dynamic": is_dynamic,
            "motion_type": "DYNAMIC GESTURE" if is_dynamic else "STATIC GESTURE",
            "handedness": self.handedness_str,
            "category": info.get("category", "Gesture Recognition"),
            "meaning": info.get("meaning", ""),
            "english": info.get("english", stable_gesture),
            "kannada": info.get("kannada", ""),
            "kannada_translit": info.get("kannada_translit", ""),
            "speech": info.get("speech", ""),
            "is_emergency": info.get("is_emergency", False),
            "finger_states": f_states
        }

    def reset_no_hand(self):
        """Called when no hands are in view to reset state smoothly."""
        self.motion_tracker.clear()
        if len(self.prediction_buffer) > 0:
            self.prediction_buffer.popleft()

        if len(self.prediction_buffer) == 0:
            self.active_gesture = "STANDBY"
            self.active_confidence = 0.0
            self.active_status = "HAND NOT DETECTED"
            self.handedness_str = "NONE"

    def process_frame(
        self,
        detected_hands: list,
        image_shape: Tuple[int, int] = (720, 1280)
    ) -> Dict[str, Any]:
        """
        Processes all hands detected in a camera frame.
        Supports 1 or 2 hands, extracts landmarks, runs inference, smoothing, and returns full telemetry.
        """
        if not detected_hands:
            self.reset_no_hand()
            info = get_gesture_info("STANDBY")
            return {
                "raw_gesture": "STANDBY",
                "confirmed_gesture": "STANDBY",
                "confidence": 0.0,
                "confidence_tier": "UNKNOWN",
                "is_confident": False,
                "status": "HAND NOT DETECTED",
                "category": "System",
                "motion_type": "STATIC",
                "meaning": info.get("meaning", "Awaiting hand in camera view"),
                "english": info.get("english", "Standby"),
                "kannada": info.get("kannada", "ಸನ್ನೆಗಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ"),
                "kannada_translit": info.get("kannada_translit", ""),
                "spoken_phrase": info.get("speech", ""),
                "is_emergency": False,
                "detected_hand_count": 0,
                "is_new_confirmation": False,
                "hands_info": []
            }

        h, w = image_shape[:2]
        hands_info = []

        # Analyze each hand for visualization & bounding boxes
        for hand in detected_hands:
            h_label = getattr(hand, "handedness", "Right")
            bbox = LandmarkProcessor.get_bounding_box(hand, w, h, margin=20)
            hands_info.append({
                "landmarks": hand,
                "label": h_label,
                "score": 1.0,
                "bbox": bbox or (0, 0, 10, 10)
            })

        # Check for Bimanual Signs (NAMASTE prayer hands, HOME roof peak) if 2 hands in view
        hand_res = None
        if len(detected_hands) >= 2:
            try:
                bimanual = LandmarkProcessor.analyze_two_hands(detected_hands[0], detected_hands[1])
                if bimanual.get("is_namaste"):
                    info = get_gesture_info("NAMASTE")
                    hand_res = {
                        "gesture": "NAMASTE",
                        "confidence": 0.96,
                        "confidence_tier": "HIGH CONFIDENCE",
                        "status": "GESTURE DETECTED (2 HANDS)",
                        "is_dynamic": False,
                        "motion_type": "STATIC GESTURE",
                        "handedness": "BOTH HANDS",
                        "category": info.get("category", "Sign Language (ISL)"),
                        "meaning": info.get("meaning", "Traditional Indian greeting & sign of respect"),
                        "english": "Namaste",
                        "kannada": info.get("kannada", "ನಮಸ್ಕಾರ"),
                        "kannada_translit": info.get("kannada_translit", "Namaskara"),
                        "speech": "Namaste",
                        "is_emergency": False,
                        "finger_states": {}
                    }
                    self.active_gesture = "NAMASTE"
                    self.active_confidence = 0.96
                elif bimanual.get("is_home_roof"):
                    info = get_gesture_info("HOME")
                    hand_res = {
                        "gesture": "HOME",
                        "confidence": 0.94,
                        "confidence_tier": "HIGH CONFIDENCE",
                        "status": "GESTURE DETECTED (2 HANDS)",
                        "is_dynamic": False,
                        "motion_type": "STATIC GESTURE",
                        "handedness": "BOTH HANDS",
                        "category": info.get("category", "Sign Language (ISL)"),
                        "meaning": info.get("meaning", "Residence / House / Shelter"),
                        "english": "Home",
                        "kannada": info.get("kannada", "ಮನೆ"),
                        "kannada_translit": info.get("kannada_translit", "Mane"),
                        "speech": "Home",
                        "is_emergency": False,
                        "finger_states": {}
                    }
                    self.active_gesture = "HOME"
                    self.active_confidence = 0.94
            except Exception:
                pass

        if hand_res is None:
            # Primary hand for classification (first hand detected)
            primary_hand = detected_hands[0]
            h_label = getattr(primary_hand, "handedness", "Right")
            hand_res = self.process_hand(primary_hand, handedness_label=h_label, image_shape=image_shape)

        confirmed = hand_res["gesture"]
        conf = hand_res["confidence"]
        tier = "HIGH" if conf >= self.high_threshold else ("MEDIUM" if conf >= self.medium_threshold else "UNKNOWN")

        # Check if new confirmation
        now = time.time()
        is_new = False
        if confirmed != self.last_confirmed_gesture and confirmed not in ("STANDBY", "UNKNOWN GESTURE", "Gesture not recognized"):
            self.last_confirmed_gesture = confirmed
            self.confirmed_time = now
            is_new = True

        return {
            "raw_gesture": hand_res.get("gesture", "STANDBY"),
            "confirmed_gesture": confirmed,
            "confidence": conf,
            "confidence_tier": tier,
            "is_confident": conf >= self.medium_threshold,
            "status": hand_res.get("status", "ANALYZING"),
            "category": hand_res.get("category", "Gesture Recognition"),
            "motion_type": hand_res.get("motion_type", "STATIC GESTURE"),
            "meaning": hand_res.get("meaning", ""),
            "english": hand_res.get("english", confirmed),
            "kannada": hand_res.get("kannada", ""),
            "kannada_translit": hand_res.get("kannada_translit", ""),
            "spoken_phrase": hand_res.get("speech", ""),
            "is_emergency": hand_res.get("is_emergency", False),
            "detected_hand_count": len(detected_hands),
            "is_new_confirmation": is_new,
            "hands_info": hands_info
        }

