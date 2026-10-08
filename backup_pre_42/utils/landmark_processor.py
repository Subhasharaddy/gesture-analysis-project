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
        Returns a (21, 3) numpy array.
        """
        coords = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            coords[i] = [lm.x, lm.y, lm.z]
        return coords

    @classmethod
    def normalize_landmarks(cls, raw_coords: np.ndarray) -> np.ndarray:
        """
        Performs wrist-relative translation and max-distance scaling.

        Args:
            raw_coords: (21, 3) array of raw coordinates.

        Returns:
            normalized_coords: (21, 3) translation and scale invariant coordinates.
        """
        wrist = raw_coords[cls.WRIST]
        # Step 1: Translate wrist to (0, 0, 0)
        translated = raw_coords - wrist

        # Step 2: Compute maximum Euclidean distance from wrist to any landmark
        distances = np.linalg.norm(translated, axis=1)
        max_dist = np.max(distances)

        # Avoid division by zero if landmarks collapse
        if max_dist < 1e-6:
            max_dist = 1.0

        # Step 3: Scale coordinates so max distance equals 1.0
        normalized = translated / max_dist
        return normalized

    @classmethod
    def compute_geometric_features(cls, normalized_coords: np.ndarray) -> np.ndarray:
        """
        Computes key geometric distance and curl features:
        - Distance from each fingertip to the wrist (5 features)
        - Distance between adjacent fingertips (4 features: thumb-index, index-mid, mid-ring, ring-pinky)
        - Thumb tip to pinky tip span (1 feature)
        Total: 10 auxiliary geometric features.
        """
        wrist = normalized_coords[cls.WRIST]
        aux_features = []

        # 1. Fingertip to wrist normalized distances
        for tip_idx in cls.FINGERTIP_INDICES:
            dist = np.linalg.norm(normalized_coords[tip_idx] - wrist)
            aux_features.append(dist)

        # 2. Adjacent fingertip distances (spread of fingers)
        for i in range(len(cls.FINGERTIP_INDICES) - 1):
            tip_a = normalized_coords[cls.FINGERTIP_INDICES[i]]
            tip_b = normalized_coords[cls.FINGERTIP_INDICES[i + 1]]
            aux_features.append(np.linalg.norm(tip_a - tip_b))

        # 3. Overall hand span (thumb tip to pinky tip)
        span = np.linalg.norm(
            normalized_coords[cls.THUMB_TIP] - normalized_coords[cls.PINKY_TIP]
        )
        aux_features.append(span)

        return np.array(aux_features, dtype=np.float32)

    @classmethod
    def extract_feature_vector(cls, hand_landmarks) -> np.ndarray:
        """
        End-to-end feature extraction pipeline for a single detected hand.
        Output vector size:
          - 21 landmarks * 3 coords = 63 normalized relative coordinates
          - 10 geometric distance/span features
          Total feature vector length = 73 features.
        """
        raw_coords = cls.extract_raw_landmarks(hand_landmarks)
        norm_coords = cls.normalize_landmarks(raw_coords)
        aux_features = cls.compute_geometric_features(norm_coords)

        # Flatten 21x3 -> 63
        flat_coords = norm_coords.flatten()

        # Concatenate 63 + 10 = 73 features
        feature_vector = np.concatenate([flat_coords, aux_features])
        return feature_vector

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
