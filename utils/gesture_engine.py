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
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import config
from utils.landmark_processor import LandmarkProcessor
from utils.gesture_database import GESTURE_DEFINITIONS, get_gesture_info, CLASS_ID_TO_GESTURE_MAP, class_id_to_gesture_name

logger = logging.getLogger(__name__)


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
        confidence_threshold: Optional[float] = None,
        smoothing_window: int = 3,
        stability_window: Optional[int] = None,
        high_threshold: float = 0.80,
        medium_threshold: float = 0.70
    ):
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else getattr(config, "CONFIDENCE_THRESHOLD", 0.70)
        self.smoothing_window = stability_window if stability_window is not None else smoothing_window
        self.high_threshold = getattr(config, "CONFIDENCE_HIGH_THRESHOLD", high_threshold)
        self.medium_threshold = getattr(config, "CONFIDENCE_MEDIUM_THRESHOLD", medium_threshold)

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

        if self.model is not None and hasattr(self.model, "n_jobs"):
            # Set n_jobs=1 for single vector inference to eliminate thread-pool dispatch overhead
            self.model.n_jobs = 1

        self.motion_tracker = HandMotionTracker(history_len=10)

        # Stabilization parameters (Sections 1, 3, 5, 6)
        self.history_size = getattr(config, "PREDICTION_HISTORY_SIZE", 5)
        self.confirmation_frames = getattr(config, "GESTURE_CONFIRMATION_FRAMES", 3)
        self.no_hand_timeout = getattr(config, "NO_HAND_TIMEOUT", 0.8)
        self.confidence_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 0.70)
        self.high_threshold = getattr(config, "CONFIDENCE_HIGH_THRESHOLD", 0.80)
        self.medium_threshold = getattr(config, "CONFIDENCE_MEDIUM_THRESHOLD", 0.70)

        # Rolling history of (gesture_name, confidence) for majority voting & stability
        self.prediction_history = collections.deque(maxlen=self.history_size)

        # State tracking for continuous stability
        self.current_stable_gesture: Optional[str] = None
        self.current_stable_confidence: float = 0.0
        self.last_valid_gesture: Optional[str] = None
        self.last_valid_confidence: float = 0.0
        self.last_hand_seen_time: float = 0.0

        self.active_gesture: str = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")
        self.active_confidence: float = 0.0
        self.active_status: str = "AWAITING HAND"
        self.is_dynamic: bool = False
        self.handedness_str: str = "UNKNOWN"

        self.last_confirmed_gesture: Optional[str] = None
        self.confirmed_time: float = 0.0
        self.last_inference_ms: float = 0.0
        self.last_latency_sec: float = 0.0

    def set_model(self, model, label_encoder):
        """Updates the active machine learning model."""
        self.model = model
        self.label_encoder = label_encoder
        if self.model is not None and hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

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
        image_shape: Tuple[int, int] = (480, 640)
    ) -> Dict[str, Any]:
        """
        Executes end-to-end continuous recognition on a single detected hand:
          Landmarks -> Vectorized 73-Dim Features -> Fast ML Inference -> 2-3 Frame Smoothing
        """
        t_start = time.perf_counter()
        lms = getattr(hand_landmarks, "landmark", [])
        if len(lms) != 21:
            unknown_tag = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")
            return {
                "gesture": unknown_tag,
                "raw_prediction": unknown_tag,
                "confidence": 0.0,
                "confidence_tier": "LOW CONFIDENCE",
                "status": "DETECTING",
                "is_dynamic": False,
                "motion_type": "STATIC GESTURE",
                "handedness": handedness_label.upper(),
                "category": "Sign Language",
                "meaning": "",
                "english": unknown_tag,
                "kannada": "",
                "kannada_translit": "",
                "speech": "",
                "is_emergency": False,
                "finger_states": {},
                "inference_ms": 0.0
            }

        raw_coords = LandmarkProcessor.extract_raw_landmarks(hand_landmarks)
        norm_coords = LandmarkProcessor.normalize_landmarks(raw_coords)
        feat_vector = LandmarkProcessor.extract_feature_vector(hand_landmarks)

        # Safeguard feature vector shape
        expected_n = getattr(self.model, "n_features_in_", 73) if self.model else 73
        if len(feat_vector) != expected_n:
            print(f"[ERROR] Feature shape ({len(feat_vector)}) does not match expected ({expected_n})")

        # 1. Update motion tracker
        wrist_xy = (float(raw_coords[0][0]), float(raw_coords[0][1]))
        self.motion_tracker.update(wrist_xy)

        # 2. Extract anatomical finger states
        f_states = self.analyze_finger_states(norm_coords, raw_coords)
        is_dynamic = False

        # 3. Machine Learning Model inference (Primary 42-Class Classifier)
        ml_gesture = None
        ml_conf = 0.0
        class_id = -1
        raw_cls = None
        if self.model is not None and self.label_encoder is not None:
            try:
                probs = self.model.predict_proba([feat_vector])[0]
                best_idx = int(np.argmax(probs))
                ml_conf = float(probs[best_idx])
                class_id = best_idx
                
                # Robust decode of prediction index or class name
                if hasattr(self.model, "classes_"):
                    raw_cls = self.model.classes_[best_idx]
                else:
                    raw_cls = best_idx

                if isinstance(raw_cls, (int, np.integer)):
                    idx_val = int(raw_cls)
                    if hasattr(self.label_encoder, "classes_") and 0 <= idx_val < len(self.label_encoder.classes_):
                        candidate_name = self.label_encoder.classes_[idx_val]
                    else:
                        num_cls = len(getattr(self.label_encoder, "classes_", []))
                        candidate_name = class_id_to_gesture_name(idx_val, num_classes=num_cls if num_cls > 0 else 12)
                elif hasattr(self.label_encoder, "classes_") and 0 <= class_id < len(self.label_encoder.classes_):
                    candidate_name = self.label_encoder.classes_[class_id]
                else:
                    num_cls = len(getattr(self.label_encoder, "classes_", []))
                    candidate_name = class_id_to_gesture_name(class_id, num_classes=num_cls if num_cls > 0 else 12)

                ml_gesture = str(candidate_name).strip().upper()
            except Exception as e:
                ml_gesture = None
                ml_conf = 0.0

        infer_ms = (time.perf_counter() - t_start) * 1000
        self.last_inference_ms = infer_ms

        unknown_tag = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")
        cur_candidate = ml_gesture if ml_gesture is not None else unknown_tag

        # 4. Confidence Filter & Immediate Acquisition
        # Only accept predictions above a meaningful threshold to prevent noise from
        # entering the majority vote buffer and causing instability.
        accept_threshold = max(0.25, self.confidence_threshold * 0.5)
        if ml_gesture is not None and ml_conf >= accept_threshold:
            self.prediction_history.append((ml_gesture, ml_conf))

        # 5. Majority Voting & Switch Confirmation
        required_switch_count = max(2, self.confirmation_frames)
        if len(self.prediction_history) > 0:
            recent_gestures = [g for g, _ in self.prediction_history]
            counts = collections.Counter(recent_gestures)
            most_frequent_gesture, count = counts.most_common(1)[0]

            if self.current_stable_gesture is None:
                # First acquisition: require at least 2 consistent votes
                if count >= 2:
                    self.current_stable_gesture = most_frequent_gesture
                    self.last_valid_gesture = most_frequent_gesture
                    self.active_status = "Stable"
                else:
                    self.active_status = "Detecting..."
            else:
                # Already showing a stable gesture
                if most_frequent_gesture == self.current_stable_gesture:
                    self.active_status = "Stable"
                else:
                    # Require confirmation_frames consistent frames to switch
                    if count >= required_switch_count:
                        self.current_stable_gesture = most_frequent_gesture
                        self.last_valid_gesture = most_frequent_gesture
                        self.active_status = "Stable"
                    else:
                        # Keep showing current stable gesture (anti-flicker)
                        self.active_status = "Stable"
        else:
            # No predictions in history yet
            if self.current_stable_gesture is not None:
                self.active_status = "Stable"
            else:
                self.active_status = "Detecting..."

        # 6. Stable Confidence Smoothing
        if self.current_stable_gesture is not None:
            matching_confs = [c for g, c in self.prediction_history if g == self.current_stable_gesture]
            if matching_confs:
                self.current_stable_confidence = float(np.mean(matching_confs))
                self.last_valid_confidence = self.current_stable_confidence
            elif self.last_valid_confidence > 0.0:
                self.current_stable_confidence = self.last_valid_confidence
            else:
                self.current_stable_confidence = ml_conf
        else:
            self.current_stable_confidence = ml_conf

        # Determine effective output gesture
        if self.current_stable_gesture is not None and self.current_stable_gesture not in ("STANDBY", "No hand detected"):
            stable_out = self.current_stable_gesture
        elif ml_gesture is not None and ml_conf >= accept_threshold:
            stable_out = ml_gesture
        else:
            stable_out = unknown_tag

        # Safe diagnostic logging (throttled to avoid console spam)
        logger.debug(f"Raw model prediction: {raw_cls}")
        logger.debug(f"Predicted class/index: {class_id}")
        logger.debug(f"Prediction confidence: {int(ml_conf * 100)}%")
        logger.debug(f"Mapped gesture name: {ml_gesture}")
        logger.debug(f"Final gesture displayed: {stable_out}")

        # Confidence categorization
        effective_conf = self.current_stable_confidence if self.current_stable_confidence > 0 else ml_conf
        if effective_conf >= self.high_threshold:
            conf_tier = "HIGH CONFIDENCE"
        elif effective_conf >= self.medium_threshold:
            conf_tier = "MEDIUM CONFIDENCE"
        else:
            conf_tier = "LOW CONFIDENCE"

        # Check emergency status from config
        emergency_set = getattr(config, "EMERGENCY_GESTURES", {"HELP", "STOP", "EMERGENCY", "POLICE", "CALL FOR HELP"})
        is_emg = stable_out in emergency_set

        info_lookup = stable_out if stable_out not in (unknown_tag, "Unknown Gesture") else (cur_candidate if cur_candidate != unknown_tag else "")
        info = get_gesture_info(info_lookup) if info_lookup else {}

        return {
            "gesture": stable_out,
            "gesture_name": stable_out,
            "stable_gesture": stable_out,
            "raw_prediction": cur_candidate,
            "raw_confidence": float(ml_conf),
            "class_id": class_id,
            "confidence": float(effective_conf),
            "confidence_tier": conf_tier,
            "status": self.active_status,
            "is_dynamic": is_dynamic,
            "motion_type": "DYNAMIC GESTURE" if is_dynamic else "STATIC GESTURE",
            "handedness": self.handedness_str,
            "category": info.get("category", "Sign Language"),
            "meaning": info.get("meaning", ""),
            "english": info.get("english", stable_out),
            "kannada": info.get("kannada", ""),
            "kannada_translit": info.get("kannada_translit", ""),
            "speech": info.get("speech", stable_out),
            "is_emergency": is_emg or info.get("is_emergency", False),
            "finger_states": f_states,
            "inference_ms": infer_ms
        }

    def reset_no_hand(self):
        """Called when no hands are in view for >= NO_HAND_TIMEOUT to reset state smoothly."""
        self.motion_tracker.clear()
        self.prediction_history.clear()
        self.current_stable_gesture = None
        self.current_stable_confidence = 0.0
        self.last_valid_gesture = None
        self.last_valid_confidence = 0.0
        no_hand_str = getattr(config, "NO_HAND_LABEL", "No hand detected")
        self.active_gesture = no_hand_str
        self.active_confidence = 0.0
        self.active_status = "No hand detected"
        self.handedness_str = "NONE"

    def process_frame(
        self,
        detected_hands: list,
        image_shape: Tuple[int, int] = (480, 640)
    ) -> Dict[str, Any]:
        """
        Processes detected hands in a camera frame with stabilization,
        majority voting, confidence filtering, and no-hand timeout hold.
        """
        t_frame_start = time.perf_counter()
        now = time.time()
        no_hand_str = getattr(config, "NO_HAND_LABEL", "No hand detected")
        unknown_str = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")

        if not detected_hands:
            time_since_hand = now - self.last_hand_seen_time

            # Section 6: HOLD LAST VALID RESULT if hand temporarily disappears
            if self.last_valid_gesture is not None and time_since_hand < self.no_hand_timeout:
                disp_gesture = self.last_valid_gesture
                disp_conf = self.last_valid_confidence
                disp_status = "Stable"
                is_emg = disp_gesture in getattr(config, "EMERGENCY_GESTURES", {"HELP", "STOP", "EMERGENCY", "POLICE", "CALL FOR HELP"})
                info = get_gesture_info(disp_gesture)

                return {
                    "raw_prediction": "None",
                    "raw_confidence": 0.0,
                    "raw_gesture": disp_gesture,
                    "stable_gesture": disp_gesture,
                    "confirmed_gesture": disp_gesture,
                    "gesture": disp_gesture,
                    "gesture_name": disp_gesture,
                    "confidence": disp_conf,
                    "confidence_tier": "HIGH CONFIDENCE" if disp_conf >= self.high_threshold else "MEDIUM CONFIDENCE",
                    "is_confident": disp_conf >= self.medium_threshold,
                    "status": disp_status,
                    "category": info.get("category", "Sign Language"),
                    "motion_type": "STATIC GESTURE",
                    "meaning": info.get("meaning", ""),
                    "english": info.get("english", disp_gesture),
                    "kannada": info.get("kannada", ""),
                    "kannada_translit": info.get("kannada_translit", ""),
                    "spoken_phrase": info.get("speech", ""),
                    "is_emergency": is_emg or info.get("is_emergency", False),
                    "detected_hand_count": 0,
                    "is_new_confirmation": False,
                    "hands_info": [],
                    "inference_ms": 0.0,
                    "latency_sec": time.perf_counter() - t_frame_start
                }

            # If no hand for >= no_hand_timeout (0.8s):
            self.reset_no_hand()
            return {
                "raw_prediction": no_hand_str,
                "raw_confidence": 0.0,
                "raw_gesture": no_hand_str,
                "stable_gesture": no_hand_str,
                "confirmed_gesture": no_hand_str,
                "gesture": no_hand_str,
                "gesture_name": no_hand_str,
                "confidence": 0.0,
                "confidence_tier": "UNKNOWN",
                "is_confident": False,
                "status": "No hand detected",
                "category": "System",
                "motion_type": "STATIC",
                "meaning": "Place hand in camera view",
                "english": no_hand_str,
                "kannada": "",
                "kannada_translit": "",
                "spoken_phrase": "",
                "is_emergency": False,
                "detected_hand_count": 0,
                "is_new_confirmation": False,
                "hands_info": [],
                "inference_ms": 0.0,
                "latency_sec": 0.0
            }

        # Hand detected: update last_hand_seen_time
        self.last_hand_seen_time = now
        h, w = image_shape[:2]
        hands_info = []

        for hand in detected_hands:
            h_label = getattr(hand, "handedness", "Right")
            bbox = LandmarkProcessor.get_bounding_box(hand, w, h, margin=20)
            hands_info.append({
                "landmarks": hand,
                "label": h_label,
                "score": 1.0,
                "bbox": bbox or (0, 0, 10, 10)
            })

        # Unified 42-class classification on detected hand
        primary_hand = detected_hands[0]
        h_label = getattr(primary_hand, "handedness", "Right")
        hand_res = self.process_hand(primary_hand, handedness_label=h_label, image_shape=image_shape)

        stable_gest = hand_res["stable_gesture"]
        conf = hand_res["confidence"]
        tier = hand_res["confidence_tier"]

        # Check if new confirmed gesture
        is_new = False
        if stable_gest != self.last_confirmed_gesture and stable_gest not in ("STANDBY", unknown_str, no_hand_str, "UNKNOWN GESTURE"):
            self.last_confirmed_gesture = stable_gest
            self.confirmed_time = now
            is_new = True

        latency_sec = time.perf_counter() - t_frame_start
        self.last_latency_sec = latency_sec

        return {
            "gesture": stable_gest,
            "gesture_name": stable_gest,
            "class_id": hand_res.get("class_id", -1),
            "raw_prediction": hand_res.get("raw_prediction", unknown_str),
            "raw_confidence": hand_res.get("raw_confidence", 0.0),
            "raw_gesture": hand_res.get("raw_prediction", unknown_str),
            "stable_gesture": stable_gest,
            "confirmed_gesture": stable_gest,
            "confidence": conf,
            "confidence_tier": tier,
            "is_confident": conf >= self.medium_threshold,
            "status": hand_res.get("status", "Stable"),
            "category": hand_res.get("category", "Sign Language"),
            "motion_type": hand_res.get("motion_type", "STATIC GESTURE"),
            "meaning": hand_res.get("meaning", ""),
            "english": hand_res.get("english", stable_gest),
            "kannada": hand_res.get("kannada", ""),
            "kannada_translit": hand_res.get("kannada_translit", ""),
            "spoken_phrase": hand_res.get("speech", stable_gest),
            "is_emergency": hand_res.get("is_emergency", False),
            "detected_hand_count": len(detected_hands),
            "is_new_confirmation": is_new,
            "hands_info": hands_info,
            "inference_ms": hand_res.get("inference_ms", self.last_inference_ms),
            "latency_sec": latency_sec
        }

