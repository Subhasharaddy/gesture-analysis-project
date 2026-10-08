"""
Algorithmic Generator for 30 Sign Language Gestures.
Synthesizes 9,000 training samples (30 gestures * 300 samples) using anatomically
distinct skeletal joint templates, Gaussian noise, 3D rotations, and distance scaling.
Allows immediate training and testing of the 30-gesture system out-of-the-box.
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from types import SimpleNamespace

import config
from utils.landmark_processor import LandmarkProcessor


def create_landmark_mock(coords_21x3: np.ndarray):
    """Duck-type wrapper mimicking a MediaPipe landmark container."""
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


def get_base_skeleton(
    thumb_ext: float = 1.0,
    index_ext: float = 1.0,
    middle_ext: float = 1.0,
    ring_ext: float = 1.0,
    pinky_ext: float = 1.0,
    thumb_dir: tuple = (-1.0, -0.6),
    angle_offset: float = 0.0,
    finger_spread: float = 1.0
) -> np.ndarray:
    """
    Parametric hand model that builds a 21-joint skeleton based on individual finger extensions.
    - ext = 1.0: fully extended upright finger
    - ext = 0.0: curled finger folded into palm
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.50, 0.82, 0.0]  # Wrist

    # Base MCP positions
    mcp_x = [0.44, 0.45, 0.50, 0.55, 0.60]
    mcp_y = [0.74, 0.62, 0.60, 0.62, 0.66]

    # --- Thumb (1, 2, 3, 4) ---
    lm[1] = [0.46, 0.77, -0.01]
    lm[2] = [mcp_x[0], mcp_y[0], -0.02]
    tx, ty = thumb_dir
    lm[3] = [lm[2][0] + tx * 0.05 * thumb_ext, lm[2][1] + ty * 0.05 * thumb_ext, -0.04]
    lm[4] = [lm[2][0] + tx * 0.11 * thumb_ext, lm[2][1] + ty * 0.11 * thumb_ext, -0.06]

    # --- 4 Fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20) ---
    finger_params = [
        (5, index_ext,  -0.05 * finger_spread, [0.44, 0.62]),
        (9, middle_ext,  0.00 * finger_spread, [0.50, 0.60]),
        (13, ring_ext,   0.04 * finger_spread, [0.55, 0.62]),
        (17, pinky_ext,  0.09 * finger_spread, [0.60, 0.66])
    ]

    for base_idx, ext, spread_dx, mcp_pos in finger_params:
        lm[base_idx] = [mcp_pos[0], mcp_pos[1], -0.01]  # MCP

        if ext > 0.5:
            # Extended straight
            lm[base_idx + 1] = [mcp_pos[0] + spread_dx * 0.3, mcp_pos[1] - 0.11, -0.02]  # PIP
            lm[base_idx + 2] = [mcp_pos[0] + spread_dx * 0.6, mcp_pos[1] - 0.20, -0.03]  # DIP
            lm[base_idx + 3] = [mcp_pos[0] + spread_dx * 1.0, mcp_pos[1] - 0.31, -0.04]  # TIP
        else:
            # Curled into palm
            lm[base_idx + 1] = [mcp_pos[0], mcp_pos[1] - 0.05, -0.03]
            lm[base_idx + 2] = [mcp_pos[0] + 0.01, mcp_pos[1] + 0.02, -0.05]
            lm[base_idx + 3] = [mcp_pos[0] + 0.01, mcp_pos[1] + 0.08, -0.06]

    if angle_offset != 0.0:
        lm = rotate_landmarks_z(lm, angle_offset)

    return lm


def build_template_for_gesture(gesture_name: str) -> np.ndarray:
    """Generates a distinctive anatomical 21-joint skeleton for any of the 30 gestures."""
    if gesture_name == "HELLO":
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.0, -0.4), finger_spread=1.3, angle_offset=5.0)

    elif gesture_name == "GOOD MORNING":
        return get_base_skeleton(0.9, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.9, -0.6), finger_spread=0.8, angle_offset=-10.0)

    elif gesture_name == "GOOD NIGHT":
        return get_base_skeleton(0.3, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.4, 0.0), finger_spread=0.5, angle_offset=45.0)

    elif gesture_name == "THANK YOU":
        return get_base_skeleton(0.8, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.6, -0.5), finger_spread=0.4, angle_offset=15.0)

    elif gesture_name == "PLEASE":
        return get_base_skeleton(0.5, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.2, -0.8), finger_spread=0.2, angle_offset=-5.0)

    elif gesture_name == "SORRY":
        # 'A' fist rotated
        return get_base_skeleton(0.2, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.2, -0.8), angle_offset=-15.0)

    elif gesture_name == "YES":
        # Fist nod
        return get_base_skeleton(0.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.1, -0.5), angle_offset=0.0)

    elif gesture_name == "NO":
        # Index & Middle extended, others curled
        return get_base_skeleton(0.1, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.2, -0.2), finger_spread=0.4, angle_offset=0.0)

    elif gesture_name == "HELP":
        # Distress tuck
        lm = get_base_skeleton(0.0, 0.0, 0.0, 0.0, 0.0)
        lm[4] = [0.52, 0.65, -0.08]  # Tucked thumb deep in palm
        return lm

    elif gesture_name == "STOP":
        # Rigid open palm, fingers held straight together
        return get_base_skeleton(0.7, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.8, -0.4), finger_spread=0.2, angle_offset=0.0)

    elif gesture_name == "COME":
        # Index beckoning
        lm = get_base_skeleton(0.0, 0.6, 0.0, 0.0, 0.0, thumb_dir=(0.1, -0.2))
        lm[8] = [0.45, 0.45, -0.08]  # Index bent forward
        return lm

    elif gesture_name == "GO":
        # Index pointing outward away
        return get_base_skeleton(0.0, 1.0, 0.0, 0.0, 0.0, thumb_dir=(0.0, -0.4), angle_offset=65.0)

    elif gesture_name == "WAIT":
        return get_base_skeleton(0.6, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.8, -0.6), finger_spread=0.9, angle_offset=0.0)

    elif gesture_name == "OK":
        # 'O' circle (Thumb tip touches Index tip), 3 fingers straight up
        lm = get_base_skeleton(0.8, 0.5, 1.0, 1.0, 1.0, thumb_dir=(0.2, -0.6), finger_spread=1.0)
        # Touch thumb and index tips
        lm[4] = [0.46, 0.52, -0.05]
        lm[8] = [0.46, 0.52, -0.05]
        return lm

    elif gesture_name == "GOOD":
        # Thumbs UP!
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.0, -1.0))
        lm[4] = [0.44, 0.45, -0.08]  # Thumb high
        return lm

    elif gesture_name == "BAD":
        # Thumbs DOWN!
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.0, 1.0))
        lm[4] = [0.44, 0.95, -0.04]  # Thumb low
        return lm

    elif gesture_name == "LOVE":
        # 'ILY' (Thumb, Index, Pinky extended, Middle & Ring curled)
        return get_base_skeleton(1.0, 1.0, 0.0, 0.0, 1.0, thumb_dir=(-1.0, -0.5), finger_spread=1.5)

    elif gesture_name == "FRIEND":
        # Index and Middle crossed / hooked
        return get_base_skeleton(0.3, 0.8, 0.8, 0.0, 0.0, thumb_dir=(0.1, -0.5), finger_spread=0.2, angle_offset=-10.0)

    elif gesture_name == "WATER":
        # 'W' sign (Index, Middle, Ring extended, Thumb holding Pinky)
        lm = get_base_skeleton(0.5, 1.0, 1.0, 1.0, 0.0, thumb_dir=(0.6, 0.0), finger_spread=0.8)
        lm[4] = [0.57, 0.68, -0.05]
        lm[20] = [0.57, 0.68, -0.05]
        return lm

    elif gesture_name == "FOOD":
        # Bunched eating fingers touching at one point
        lm = get_base_skeleton(0.5, 0.5, 0.5, 0.5, 0.5)
        pinch_pt = [0.48, 0.52, -0.08]
        lm[4] = pinch_pt
        lm[8] = pinch_pt
        lm[12] = [pinch_pt[0] + 0.01, pinch_pt[1], pinch_pt[2]]
        lm[16] = [pinch_pt[0] + 0.01, pinch_pt[1] + 0.01, pinch_pt[2]]
        lm[20] = [pinch_pt[0] + 0.02, pinch_pt[1] + 0.01, pinch_pt[2]]
        return lm

    elif gesture_name == "HOME":
        # Roof peak bunched fingers
        return get_base_skeleton(0.6, 0.7, 0.7, 0.7, 0.7, thumb_dir=(-0.2, -0.6), finger_spread=0.2, angle_offset=0.0)

    elif gesture_name == "SCHOOL":
        # Flat horizontal clap
        return get_base_skeleton(0.8, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.6, -0.3), finger_spread=0.3, angle_offset=80.0)

    elif gesture_name == "HOSPITAL":
        # 'H' shape (Index & Middle straight together horizontal)
        return get_base_skeleton(0.2, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.1, -0.4), finger_spread=0.1, angle_offset=45.0)

    elif gesture_name == "DOCTOR":
        # Two fingers tapping wrist pulse
        lm = get_base_skeleton(0.2, 0.8, 0.8, 0.0, 0.0, thumb_dir=(0.0, -0.2), angle_offset=-40.0)
        lm[8] = [0.48, 0.75, -0.03]
        lm[12] = [0.50, 0.76, -0.03]
        return lm

    elif gesture_name == "POLICE":
        # 'C' cup shape (curved fingers)
        lm = get_base_skeleton(0.8, 0.7, 0.7, 0.7, 0.7, thumb_dir=(-0.7, 0.3), finger_spread=0.3)
        lm[8][2] = -0.10
        lm[12][2] = -0.10
        return lm

    elif gesture_name == "EMERGENCY":
        # Flashing wide distress hand
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.2, -0.3), finger_spread=1.6, angle_offset=25.0)

    elif gesture_name == "PHONE":
        # 'Y' telephone hand (Thumb and Pinky extended far, 3 inner fingers curled)
        return get_base_skeleton(1.0, 0.0, 0.0, 0.0, 1.0, thumb_dir=(-1.1, -0.3), finger_spread=1.8, angle_offset=-15.0)

    elif gesture_name == "THANKS":
        # Flat palm forward offering
        return get_base_skeleton(0.7, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.5, -0.6), finger_spread=0.5, angle_offset=-25.0)

    elif gesture_name == "I AM FINE":
        # '5' hand with thumb spread
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.9, -0.5), finger_spread=1.2, angle_offset=0.0)

    elif gesture_name == "CALL FOR HELP":
        # Urgent telephone distress (Y hand tilted urgently)
        return get_base_skeleton(1.0, 0.0, 0.0, 0.0, 1.0, thumb_dir=(-1.2, -0.4), finger_spread=1.9, angle_offset=30.0)

    else:
        # Fallback open palm
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0)


def generate_30_gesture_dataset(samples_per_gesture: int = 300):
    print("=" * 70)
    print("GENERATING 30-GESTURE DATASET (30 classes x 300 = 9,000 samples)...")
    print("=" * 70)

    feature_names = LandmarkProcessor.get_feature_names()
    columns = ["label"] + feature_names
    dataset_rows = []

    np.random.seed(42)

    for idx, gesture in enumerate(config.GESTURES, 1):
        print(f"[{idx:2d}/30] Synthesizing {samples_per_gesture} samples for '{gesture}'...")
        template = build_template_for_gesture(gesture)

        for _ in range(samples_per_gesture):
            # 1. Random 2D in-plane rotation (-15 to +15 deg)
            angle = np.random.uniform(-15.0, 15.0)
            aug_lm = rotate_landmarks_z(template.copy(), angle)

            # 2. Gaussian joint jitter
            noise = np.random.normal(0.0, 0.010, aug_lm.shape)
            noise[0] = 0.0  # Anchor wrist
            aug_lm += noise

            # 3. Distance / scale scaling (0.75x to 1.25x)
            scale = np.random.uniform(0.75, 1.25)
            wrist = aug_lm[0].copy()
            aug_lm = (aug_lm - wrist) * scale + wrist

            # 4. Translation shifts
            aug_lm[:, 0] += np.random.uniform(-0.12, 0.12)
            aug_lm[:, 1] += np.random.uniform(-0.12, 0.12)

            # Feature extraction
            mock_hand = create_landmark_mock(aug_lm)
            feat_vec = LandmarkProcessor.extract_feature_vector(mock_hand)
            dataset_rows.append([gesture] + feat_vec.tolist())

    df = pd.DataFrame(dataset_rows, columns=columns)
    # Shuffle entire dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    config.DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_CSV_PATH, index=False)

    print("=" * 70)
    print(f"[SUCCESS] Generated {len(df)} total samples across {len(config.GESTURES)} gestures.")
    print(f"Dataset saved to: {config.DATASET_CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    generate_30_gesture_dataset()

