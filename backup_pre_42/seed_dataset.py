"""
Seeds the dataset directory with 300 baseline samples per gesture:
  dataset/
    HELLO/
    YES/
    NO/
    HELP/
    STOP/
"""

import numpy as np
import math
from pathlib import Path
from types import SimpleNamespace

import config
from utils.landmark_processor import LandmarkProcessor


def create_landmark_mock(coords_21x3: np.ndarray):
    landmarks_list = [SimpleNamespace(x=row[0], y=row[1], z=row[2]) for row in coords_21x3]
    return SimpleNamespace(landmark=landmarks_list)


def rotate_landmarks_z(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=np.float32)
    wrist = coords[0]
    translated = coords - wrist
    rotated = np.dot(translated, rot_matrix.T) + wrist
    return rotated


def build_template_stop():
    """STOP: Open palm facing camera."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    lm[1] = [0.44, 0.78, -0.02]; lm[2] = [0.38, 0.70, -0.04]; lm[3] = [0.34, 0.62, -0.06]; lm[4] = [0.30, 0.55, -0.07]
    lm[5] = [0.44, 0.60, -0.02]; lm[6] = [0.43, 0.48, -0.03]; lm[7] = [0.42, 0.38, -0.04]; lm[8] = [0.42, 0.28, -0.05]
    lm[9] = [0.50, 0.58, -0.01]; lm[10] = [0.50, 0.45, -0.02]; lm[11] = [0.50, 0.34, -0.03]; lm[12] = [0.50, 0.23, -0.04]
    lm[13] = [0.56, 0.60, 0.0]; lm[14] = [0.57, 0.48, 0.0]; lm[15] = [0.58, 0.38, 0.0]; lm[16] = [0.58, 0.29, 0.0]
    lm[17] = [0.62, 0.65, 0.02]; lm[18] = [0.64, 0.55, 0.03]; lm[19] = [0.65, 0.47, 0.04]; lm[20] = [0.66, 0.40, 0.05]
    return lm


def build_template_hello():
    """HELLO: Waving open palm."""
    lm = build_template_stop()
    lm[4] = [0.27, 0.58, -0.06]
    lm[8] = [0.40, 0.27, -0.04]
    lm[12] = [0.51, 0.22, -0.03]
    lm[16] = [0.61, 0.28, 0.01]
    lm[20] = [0.70, 0.37, 0.06]
    return lm


def build_template_yes():
    """YES: Fist nod."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    lm[1] = [0.44, 0.78, -0.03]; lm[2] = [0.40, 0.72, -0.06]; lm[3] = [0.44, 0.66, -0.08]; lm[4] = [0.49, 0.64, -0.09]
    lm[5] = [0.45, 0.65, -0.02]; lm[6] = [0.45, 0.58, -0.05]; lm[7] = [0.46, 0.66, -0.08]; lm[8] = [0.47, 0.72, -0.06]
    lm[9] = [0.50, 0.64, -0.01]; lm[10] = [0.50, 0.57, -0.04]; lm[11] = [0.51, 0.65, -0.07]; lm[12] = [0.51, 0.73, -0.05]
    lm[13] = [0.55, 0.65, 0.0]; lm[14] = [0.55, 0.58, -0.03]; lm[15] = [0.55, 0.66, -0.06]; lm[16] = [0.55, 0.73, -0.04]
    lm[17] = [0.60, 0.68, 0.02]; lm[18] = [0.60, 0.61, -0.01]; lm[19] = [0.60, 0.68, -0.04]; lm[20] = [0.59, 0.75, -0.02]
    return lm


def build_template_no():
    """NO: Index and Middle upright together, others curled."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    lm[1] = [0.44, 0.78, -0.02]; lm[2] = [0.42, 0.73, -0.05]; lm[3] = [0.45, 0.68, -0.07]; lm[4] = [0.48, 0.67, -0.08]
    lm[5] = [0.46, 0.62, -0.02]; lm[6] = [0.46, 0.50, -0.03]; lm[7] = [0.47, 0.38, -0.04]; lm[8] = [0.47, 0.28, -0.05]
    lm[9] = [0.51, 0.61, -0.01]; lm[10] = [0.51, 0.49, -0.02]; lm[11] = [0.51, 0.37, -0.03]; lm[12] = [0.51, 0.27, -0.04]
    lm[13] = [0.56, 0.65, 0.01]; lm[14] = [0.56, 0.59, -0.03]; lm[15] = [0.56, 0.67, -0.06]; lm[16] = [0.55, 0.74, -0.04]
    lm[17] = [0.61, 0.68, 0.03]; lm[18] = [0.61, 0.62, 0.0]; lm[19] = [0.60, 0.69, -0.03]; lm[20] = [0.59, 0.75, -0.02]
    return lm


def build_template_help():
    """HELP: Thumb folded into palm, fingers wrapped over."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    lm[1] = [0.45, 0.79, -0.02]; lm[2] = [0.45, 0.72, -0.05]; lm[3] = [0.48, 0.67, -0.07]; lm[4] = [0.52, 0.65, -0.08]
    lm[5] = [0.44, 0.62, -0.01]; lm[6] = [0.44, 0.52, -0.04]; lm[7] = [0.46, 0.57, -0.08]; lm[8] = [0.48, 0.64, -0.10]
    lm[9] = [0.49, 0.61, 0.0]; lm[10] = [0.50, 0.51, -0.03]; lm[11] = [0.51, 0.56, -0.07]; lm[12] = [0.52, 0.63, -0.09]
    lm[13] = [0.55, 0.63, 0.01]; lm[14] = [0.55, 0.53, -0.02]; lm[15] = [0.55, 0.58, -0.06]; lm[16] = [0.55, 0.65, -0.08]
    lm[17] = [0.60, 0.67, 0.03]; lm[18] = [0.60, 0.58, 0.0]; lm[19] = [0.60, 0.63, -0.04]; lm[20] = [0.59, 0.68, -0.06]
    return lm


def seed_all_gestures(samples_per_class: int = 300):
    print("=" * 60)
    print("SEEDING DATASET (300 SAMPLES PER GESTURE)")
    print(f"Target Directory: {config.DATASET_DIR}")
    print("=" * 60)

    templates = {
        "HELLO": build_template_hello(),
        "YES":   build_template_yes(),
        "NO":    build_template_no(),
        "HELP":  build_template_help(),
        "STOP":  build_template_stop()
    }

    np.random.seed(42)

    for g, base_template in templates.items():
        folder = config.DATASET_DIR / g
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Seeding '{g}' -> {folder} ...")

        for i in range(1, samples_per_class + 1):
            # 1. Rotation variation
            angle = np.random.uniform(-18.0, 18.0)
            aug_lm = rotate_landmarks_z(base_template.copy(), angle)

            # 2. Joint jitter
            noise = np.random.normal(0.0, 0.012, aug_lm.shape)
            noise[0] = 0.0
            aug_lm += noise

            # 3. Distance scale factor
            scale = np.random.uniform(0.80, 1.20)
            wrist = aug_lm[0].copy()
            aug_lm = (aug_lm - wrist) * scale + wrist

            # 4. Translation
            aug_lm[:, 0] += np.random.uniform(-0.10, 0.10)
            aug_lm[:, 1] += np.random.uniform(-0.10, 0.10)

            # Extract 73D vector
            mock = create_landmark_mock(aug_lm)
            feat_vec = LandmarkProcessor.extract_feature_vector(mock)

            sample_file = folder / f"sample_{i:04d}.npy"
            np.save(str(sample_file), feat_vec)

    print("\n[SUCCESS] Seeded 1,500 total samples across all 5 gestures.")


if __name__ == "__main__":
    seed_all_gestures()
