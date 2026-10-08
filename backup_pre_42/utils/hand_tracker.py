"""
Universal Hand Tracker Module.
Supports both MediaPipe 1.x (HandLandmarker Tasks API) and MediaPipe 0.x (mp.solutions.hands)
with native OpenCV landmark and connection drawing routines.
"""

import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace
import logging

import config

logger = logging.getLogger("HandTracker")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]


class UniversalHandTracker:
    """
    Unified Hand Landmark Detector and Renderer.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = Path(__file__).resolve().parent.parent / "hand_landmarker.task"

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.mode = None
        self.detector = None

        self._initialize_detector()

    def _ensure_task_model(self) -> str:
        """Downloads hand_landmarker.task if not present."""
        if not self.MODEL_PATH.exists():
            logger.info("Downloading MediaPipe hand_landmarker.task model asset...")
            urllib.request.urlretrieve(self.MODEL_URL, str(self.MODEL_PATH))
            logger.info("hand_landmarker.task downloaded successfully.")
        return str(self.MODEL_PATH)

    def _initialize_detector(self):
        # 1. Try MediaPipe 1.x Tasks API
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_file = self._ensure_task_model()
            base_options = python.BaseOptions(model_asset_path=task_file)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.mode = "TASKS_API"
            logger.info("Initialized MediaPipe 1.x HandLandmarker (Tasks API).")
            return
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe Tasks API: {e}. Trying legacy solutions...")

        # 2. Try MediaPipe 0.x Solutions API
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence
                )
                self.mode = "SOLUTIONS_API"
                logger.info("Initialized MediaPipe 0.x Solutions API.")
                return
        except Exception as e2:
            logger.error(f"Could not initialize MediaPipe Solutions API: {e2}")

        self.mode = "NONE"
        logger.error("No valid MediaPipe detector backend found!")

    def process(self, frame_bgr: np.ndarray):
        """
        Processes a BGR image frame and returns detected hands with landmarks and handedness.
        """
        if self.mode == "TASKS_API":
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            detected_hands = []
            if detection_result.hand_landmarks:
                for i, hand_lms in enumerate(detection_result.hand_landmarks):
                    lms_list = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z) for lm in hand_lms]
                    # Extract handedness if available
                    handedness_str = "Right"
                    if detection_result.handedness and i < len(detection_result.handedness):
                        if len(detection_result.handedness[i]) > 0:
                            handedness_str = detection_result.handedness[i][0].category_name

                    detected_hands.append(SimpleNamespace(
                        landmark=lms_list,
                        handedness=handedness_str
                    ))
            return detected_hands

        elif self.mode == "SOLUTIONS_API":
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                detected_hands = []
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    handedness_str = "Right"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness_str = results.multi_handedness[i].classification[0].label
                    hand_obj = SimpleNamespace(
                        landmark=hand_lms.landmark,
                        handedness=handedness_str
                    )
                    detected_hands.append(hand_obj)
                return detected_hands
            return []

        return []

    @staticmethod
    def draw_landmarks(frame: np.ndarray, hand_landmarks, handedness_label: str = ""):
        """
        Draws high-contrast, beautiful hand skeletal lines and joint circles directly with OpenCV.
        """
        h, w = frame.shape[:2]
        pts = []
        for lm in hand_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))

        # Draw bone connections
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = pts[p1_idx]
            pt2 = pts[p2_idx]
            cv2.line(frame, pt1, pt2, (235, 206, 0), 2, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(pts):
            radius = 4 if i not in [4, 8, 12, 16, 20] else 6  # Fingertips slightly larger
            joint_color = (0, 255, 120) if i in [4, 8, 12, 16, 20] else (255, 255, 255)
            cv2.circle(frame, pt, radius, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (20, 24, 33), 1, cv2.LINE_AA)

        # Optional Handedness Tag at Wrist
        h_label = handedness_label or getattr(hand_landmarks, "handedness", "")
        if h_label and len(pts) > 0:
            wx, wy = pts[0]
            cv2.putText(
                frame,
                h_label.upper(),
                (wx - 20, min(h - 10, wy + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 210, 255),
                1,
                cv2.LINE_AA
            )

    def close(self):
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()
