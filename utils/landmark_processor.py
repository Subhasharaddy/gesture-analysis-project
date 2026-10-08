"""
Landmark Processing and Feature Extraction Module.
Transforms raw MediaPipe 3D hand landmarks into translation-invariant,
scale-invariant, and rotation-robust feature vectors for machine learning.
"""

import numpy as np
import math


class LandmarkProcessor:
    """
    Extracts and normalizes hand landmark coordinates to make the ML model
    invariant to hand position (translation) and hand distance (scale).
    """

    NUM_LANDMARKS = 21

    # Finger landmark indices in MediaPipe
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    @staticmethod
    def extract_raw_landmarks(hand_landmarks) -> np.ndarray:
        """
        Extracts raw (x, y, z) coordinates from a MediaPipe hand_landmarks object.
        Returns a (21, 3) numpy array using fast vectorized construction.
        """
        return np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)

    @classmethod
    def normalize_landmarks(cls, raw_coords: np.ndarray) -> np.ndarray:
        """
        Performs wrist-relative translation and max-distance scaling.
        Vectorized for maximum speed.
        """
        translated = raw_coords - raw_coords[cls.WRIST]
        sq_dists = np.sum(translated ** 2, axis=1)
        max_dist = float(np.sqrt(np.max(sq_dists)))

        if max_dist < 1e-6:
            max_dist = 1.0

        return translated / max_dist

    @classmethod
    def compute_geometric_features(cls, normalized_coords: np.ndarray) -> np.ndarray:
        """
        Computes 10 key geometric distance and curl features via fast vectorized NumPy operations:
        - Distance from each fingertip to the wrist (5 features)
        - Distance between adjacent fingertips (4 features)
        - Thumb tip to pinky tip span (1 feature)
        """
        tips = normalized_coords[cls.FINGERTIP_INDICES]  # shape (5, 3)
        wrist_dists = np.sqrt(np.sum(tips ** 2, axis=1))  # (5,)

        adj_diff = tips[:-1] - tips[1:]  # (4, 3)
        adj_dists = np.sqrt(np.sum(adj_diff ** 2, axis=1))  # (4,)

        span = np.sqrt(np.sum((tips[0] - tips[4]) ** 2, keepdims=True))  # (1,)

        return np.concatenate([wrist_dists, adj_dists, span], axis=0).astype(np.float32)

    @classmethod
    def extract_feature_vector(cls, hand_landmarks) -> np.ndarray:
        """
        Fast end-to-end 73-dimensional feature extraction for a single detected hand:
          - 21 normalized 3D relative coords = 63 features
          - 10 vectorized geometric distance/span features
          Total = 73 features.
        """
        raw_coords = cls.extract_raw_landmarks(hand_landmarks)
        norm_coords = cls.normalize_landmarks(raw_coords)
        aux_features = cls.compute_geometric_features(norm_coords)
        return np.concatenate([norm_coords.ravel(), aux_features])

    @classmethod
    def get_feature_names(cls) -> list:
        """Returns descriptive names for each feature in the 73-dimensional vector."""
        names = []
        for i in range(cls.NUM_LANDMARKS):
            names.extend([f"lm_{i}_norm_x", f"lm_{i}_norm_y", f"lm_{i}_norm_z"])

        names.extend([
            "dist_wrist_thumb",
            "dist_wrist_index",
            "dist_wrist_middle",
            "dist_wrist_ring",
            "dist_wrist_pinky",
            "dist_thumb_index",
            "dist_index_middle",
            "dist_middle_ring",
            "dist_ring_pinky",
            "hand_span_thumb_pinky"
        ])
        return names

    @staticmethod
    def get_bounding_box(hand_landmarks, img_w: int, img_h: int, margin: int = 25):
        """
        Calculates pixel bounding box coordinates for drawing around hand on UI.
        Returns: (x_min, y_min, x_max, y_max)
        """
        xs = [int(lm.x * img_w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * img_h) for lm in hand_landmarks.landmark]

        x_min = max(0, min(xs) - margin)
        y_min = max(0, min(ys) - margin)
        x_max = min(img_w, max(xs) + margin)
        y_max = min(img_h, max(ys) + margin)
        return x_min, y_min, x_max, y_max

    @classmethod
    def analyze_two_hands(cls, hand1, hand2) -> dict:
        """
        Computes spatial metrics between two detected hands to detect bimanual signs (NAMASTE, HOME).
        """
        raw1 = cls.extract_raw_landmarks(hand1)
        raw2 = cls.extract_raw_landmarks(hand2)

        wrist_dist = float(np.linalg.norm(raw1[cls.WRIST] - raw2[cls.WRIST]))
        middle_tip_dist = float(np.linalg.norm(raw1[cls.MIDDLE_TIP] - raw2[cls.MIDDLE_TIP]))
        index_tip_dist = float(np.linalg.norm(raw1[cls.INDEX_TIP] - raw2[cls.INDEX_TIP]))

        # Check vertical orientation
        wrist1_y = raw1[cls.WRIST][1]
        wrist2_y = raw2[cls.WRIST][1]
        tip1_y = raw1[cls.MIDDLE_TIP][1]
        tip2_y = raw2[cls.MIDDLE_TIP][1]

        both_upright = (tip1_y < wrist1_y) and (tip2_y < wrist2_y)
        
        # Prayer pose (NAMASTE): Both wrists close, tips close, hands upright
        is_namaste = both_upright and (wrist_dist < 0.25) and (middle_tip_dist < 0.14)
        
        # Roof shape (HOME): Tips touching at the top, wrists separated at the bottom
        is_home_roof = both_upright and (wrist_dist > 0.16) and (middle_tip_dist < 0.15)

        return {
            "wrist_dist": wrist_dist,
            "middle_tip_dist": middle_tip_dist,
            "both_upright": both_upright,
            "is_namaste": is_namaste,
            "is_home_roof": is_home_roof
        }
