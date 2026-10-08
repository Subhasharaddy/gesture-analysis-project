"""
Baseline Dataset Generator for Exhibition Out-Of-The-Box Execution.
Synthesizes anatomically accurate 21-landmark hand skeletal configurations
for HELLO, YES, NO, HELP, and STOP with realistic noise and rotation augmentations.
Ensures students and evaluators can immediately train and run the system.
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from types import SimpleNamespace

import config
from utils.landmark_processor import LandmarkProcessor


def create_landmark_mock(coords_21x3: np.ndarray):
    """Wraps (21, 3) numpy array into MediaPipe landmark object duck-type."""
    landmarks_list = [SimpleNamespace(x=row[0], y=row[1], z=row[2]) for row in coords_21x3]
    return SimpleNamespace(landmark=landmarks_list)


def rotate_landmarks_z(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates landmarks in the XY plane around wrist (landmark 0)."""
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


def build_template_open_palm():
    """Template for OPEN PALM / STOP: All fingers upright and extended."""
    # 21 landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]    # Wrist

    # Thumb
    lm[1] = [0.44, 0.78, -0.02]
    lm[2] = [0.38, 0.70, -0.04]
    lm[3] = [0.34, 0.62, -0.06]
    lm[4] = [0.30, 0.55, -0.07] # Thumb tip extended left

    # Index
    lm[5] = [0.44, 0.60, -0.02]
    lm[6] = [0.43, 0.48, -0.03]
    lm[7] = [0.42, 0.38, -0.04]
    lm[8] = [0.42, 0.28, -0.05] # Index tip high

    # Middle
    lm[9] = [0.50, 0.58, -0.01]
    lm[10] = [0.50, 0.45, -0.02]
    lm[11] = [0.50, 0.34, -0.03]
    lm[12] = [0.50, 0.23, -0.04] # Middle tip highest

    # Ring
    lm[13] = [0.56, 0.60, 0.0]
    lm[14] = [0.57, 0.48, 0.0]
    lm[15] = [0.58, 0.38, 0.0]
    lm[16] = [0.58, 0.29, 0.0]  # Ring tip high

    # Pinky
    lm[17] = [0.62, 0.65, 0.02]
    lm[18] = [0.64, 0.55, 0.03]
    lm[19] = [0.65, 0.47, 0.04]
    lm[20] = [0.66, 0.40, 0.05] # Pinky tip high
    return lm


def build_template_hello():
    """Template for HELLO: Open palm tilted slightly waving outward, thumb relaxed."""
    lm = build_template_open_palm()
    # Tilt slightly in Z and widen finger spread
    lm[4] = [0.27, 0.58, -0.06]
    lm[8] = [0.40, 0.27, -0.04]
    lm[12] = [0.51, 0.22, -0.03]
    lm[16] = [0.61, 0.28, 0.01]
    lm[20] = [0.70, 0.37, 0.06]
    return lm


def build_template_yes():
    """Template for YES (Fist nod): All 4 fingers curled down tightly, thumb folded across."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]    # Wrist

    # Thumb folded over curled fingers
    lm[1] = [0.44, 0.78, -0.03]
    lm[2] = [0.40, 0.72, -0.06]
    lm[3] = [0.44, 0.66, -0.08]
    lm[4] = [0.49, 0.64, -0.09]

    # Index curled in
    lm[5] = [0.45, 0.65, -0.02]
    lm[6] = [0.45, 0.58, -0.05]
    lm[7] = [0.46, 0.66, -0.08]
    lm[8] = [0.47, 0.72, -0.06] # Curled near palm

    # Middle curled in
    lm[9] = [0.50, 0.64, -0.01]
    lm[10] = [0.50, 0.57, -0.04]
    lm[11] = [0.51, 0.65, -0.07]
    lm[12] = [0.51, 0.73, -0.05]

    # Ring curled in
    lm[13] = [0.55, 0.65, 0.0]
    lm[14] = [0.55, 0.58, -0.03]
    lm[15] = [0.55, 0.66, -0.06]
    lm[16] = [0.55, 0.73, -0.04]

    # Pinky curled in
    lm[17] = [0.60, 0.68, 0.02]
    lm[18] = [0.60, 0.61, -0.01]
    lm[19] = [0.60, 0.68, -0.04]
    lm[20] = [0.59, 0.75, -0.02]
    return lm


def build_template_no():
    """Template for NO: Index and Middle extended upright together, Ring/Pinky curled, Thumb tucked."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]    # Wrist

    # Thumb resting against palm
    lm[1] = [0.44, 0.78, -0.02]
    lm[2] = [0.42, 0.73, -0.05]
    lm[3] = [0.45, 0.68, -0.07]
    lm[4] = [0.48, 0.67, -0.08]

    # Index straight up
    lm[5] = [0.46, 0.62, -0.02]
    lm[6] = [0.46, 0.50, -0.03]
    lm[7] = [0.47, 0.38, -0.04]
    lm[8] = [0.47, 0.28, -0.05]

    # Middle straight up right next to index
    lm[9] = [0.51, 0.61, -0.01]
    lm[10] = [0.51, 0.49, -0.02]
    lm[11] = [0.51, 0.37, -0.03]
    lm[12] = [0.51, 0.27, -0.04]

    # Ring curled into palm
    lm[13] = [0.56, 0.65, 0.01]
    lm[14] = [0.56, 0.59, -0.03]
    lm[15] = [0.56, 0.67, -0.06]
    lm[16] = [0.55, 0.74, -0.04]

    # Pinky curled into palm
    lm[17] = [0.61, 0.68, 0.03]
    lm[18] = [0.61, 0.62, 0.0]
    lm[19] = [0.60, 0.69, -0.03]
    lm[20] = [0.59, 0.75, -0.02]
    return lm


def build_template_help():
    """Template for HELP (Universal Distress / Emergency Sign):
       Thumb folded across center palm with 4 fingers wrapped over or high urgent upright signal.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]    # Wrist

    # Thumb tucked deep into center of palm
    lm[1] = [0.45, 0.79, -0.02]
    lm[2] = [0.45, 0.72, -0.05]
    lm[3] = [0.48, 0.67, -0.07]
    lm[4] = [0.52, 0.65, -0.08] # Tucked under fingers

    # 4 fingers bent at PIP/DIP covering the tucked thumb
    # Index
    lm[5] = [0.44, 0.62, -0.01]
    lm[6] = [0.44, 0.52, -0.04]
    lm[7] = [0.46, 0.57, -0.08]
    lm[8] = [0.48, 0.64, -0.10]

    # Middle
    lm[9] = [0.49, 0.61, 0.0]
    lm[10] = [0.50, 0.51, -0.03]
    lm[11] = [0.51, 0.56, -0.07]
    lm[12] = [0.52, 0.63, -0.09]

    # Ring
    lm[13] = [0.55, 0.63, 0.01]
    lm[14] = [0.55, 0.53, -0.02]
    lm[15] = [0.55, 0.58, -0.06]
    lm[16] = [0.55, 0.65, -0.08]

    # Pinky
    lm[17] = [0.60, 0.67, 0.03]
    lm[18] = [0.60, 0.58, 0.0]
    lm[19] = [0.60, 0.63, -0.04]
    lm[20] = [0.59, 0.68, -0.06]
    return lm


def generate_baseline_dataset(samples_per_class: int = 300):
    print("=" * 65)
    print("GENERATING SYNTHETIC BASELINE SIGN DATASET...")
    print(f"Target Gestures: {config.GESTURES}")
    print(f"Samples per gesture: {samples_per_class}")
    print("=" * 65)

    templates = {
        "HELLO": build_template_hello(),
        "YES":   build_template_yes(),
        "NO":    build_template_no(),
        "HELP":  build_template_help(),
        "STOP":  build_template_open_palm()
    }

    feature_names = LandmarkProcessor.get_feature_names()
    columns = ["label"] + feature_names
    dataset_rows = []

    np.random.seed(42)

    for gesture_name, base_template in templates.items():
        print(f"Generating {samples_per_class} augmented samples for '{gesture_name}'...")
        for _ in range(samples_per_class):
            # 1. Random 2D/3D in-plane rotation between -18 and +18 degrees
            angle = np.random.uniform(-18.0, 18.0)
            aug_lm = rotate_landmarks_z(base_template.copy(), angle)

            # 2. Random anatomical joint jitter (Gaussian noise)
            noise = np.random.normal(0.0, 0.012, aug_lm.shape)
            noise[0] = 0.0  # Keep wrist anchored
            aug_lm += noise

            # 3. Random scale scaling (simulating camera distance)
            scale_factor = np.random.uniform(0.75, 1.25)
            wrist = aug_lm[0].copy()
            aug_lm = (aug_lm - wrist) * scale_factor + wrist

            # 4. Random frame translation
            shift_x = np.random.uniform(-0.15, 0.15)
            shift_y = np.random.uniform(-0.15, 0.15)
            aug_lm[:, 0] += shift_x
            aug_lm[:, 1] += shift_y

            # Extract full 73-dimensional invariant feature vector
            mock_hand = create_landmark_mock(aug_lm)
            feat_vec = LandmarkProcessor.extract_feature_vector(mock_hand)
            dataset_rows.append([gesture_name] + feat_vec.tolist())

    df = pd.DataFrame(dataset_rows, columns=columns)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    config.DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_CSV_PATH, index=False)
    print("=" * 65)
    print(f"[SUCCESS] Generated {len(df)} total samples across {len(config.GESTURES)} gestures.")
    print(f"Dataset saved to: {config.DATASET_CSV_PATH}")
    print("Class distribution:")
    print(df["label"].value_counts().to_string())
    print("=" * 65)


if __name__ == "__main__":
    generate_baseline_dataset()
