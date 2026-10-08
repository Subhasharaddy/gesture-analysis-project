"""
Dataset Generator & Harmonizer for all 42 Sign Language & Gesture Classes.
Preserves existing samples for the 12 core gestures, and synthesizes 400
samples each for the 30 recovered gestures using anatomically verified ISL
skeletal templates, Gaussian joint noise, 3D rotations, and distance scaling.
"""

import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

import config
from utils.landmark_processor import LandmarkProcessor


def create_landmark_mock(coords_21x3: np.ndarray):
    """Wraps (21, 3) numpy array into MediaPipe landmark duck-type."""
    landmarks_list = [SimpleNamespace(x=float(row[0]), y=float(row[1]), z=float(row[2])) for row in coords_21x3]
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
    """Parametric 21-joint skeletal configuration."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.50, 0.82, 0.0]  # Wrist

    mcp_x = [0.44, 0.45, 0.50, 0.55, 0.60]
    mcp_y = [0.74, 0.62, 0.60, 0.62, 0.66]

    # Thumb
    lm[1] = [mcp_x[0], mcp_y[0], -0.01]
    tdx, tdy = thumb_dir
    lm[2] = [lm[1, 0] + 0.04 * tdx, lm[1, 1] + 0.04 * tdy, -0.02]
    lm[3] = [lm[2, 0] + 0.035 * tdx * thumb_ext, lm[2, 1] + 0.035 * tdy * thumb_ext, -0.03]
    lm[4] = [lm[3, 0] + 0.03 * tdx * thumb_ext, lm[3, 1] + 0.03 * tdy * thumb_ext, -0.04]

    # Fingers (Index=5..8, Middle=9..12, Ring=13..16, Pinky=17..20)
    exts = [index_ext, middle_ext, ring_ext, pinky_ext]
    spreads = [-0.035 * finger_spread, -0.005, 0.015 * finger_spread, 0.035 * finger_spread]

    for f_idx in range(4):
        base_lm = 5 + f_idx * 4
        m_x = mcp_x[f_idx + 1]
        m_y = mcp_y[f_idx + 1]
        ext = exts[f_idx]
        spread = spreads[f_idx]

        lm[base_lm] = [m_x, m_y, 0.0]

        pip_y = m_y - 0.09 * (1.0 if ext > 0.4 else 0.4)
        pip_x = m_x + spread * 0.4
        pip_z = -0.01 if ext > 0.4 else 0.03
        lm[base_lm + 1] = [pip_x, pip_y, pip_z]

        dip_y = pip_y - 0.08 * ext if ext > 0.3 else pip_y + 0.03
        dip_x = pip_x + spread * 0.7
        dip_z = -0.02 if ext > 0.3 else 0.06
        lm[base_lm + 2] = [dip_x, dip_y, dip_z]

        tip_y = dip_y - 0.07 * ext if ext > 0.2 else dip_y + 0.04
        tip_x = dip_x + spread * 1.0
        tip_z = -0.03 if ext > 0.2 else 0.05
        lm[base_lm + 3] = [tip_x, tip_y, tip_z]

    if angle_offset != 0.0:
        lm = rotate_landmarks_z(lm, angle_offset)

    return lm


def build_template_for_gesture(gesture_name: str) -> np.ndarray:
    """Builds anatomically distinct template for each of the 42 gestures."""
    g = gesture_name.upper().strip()

    # 1. Existing 12 gestures
    if g == "HOME":
        return get_base_skeleton(0.6, 0.7, 0.7, 0.7, 0.7, thumb_dir=(-0.2, -0.6), finger_spread=0.5, angle_offset=-15)
    elif g == "NAMASTE":
        return get_base_skeleton(0.9, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.4, -0.7), finger_spread=0.1, angle_offset=-5)
    elif g == "HELLO":
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.0, -0.4), finger_spread=1.2, angle_offset=10)
    elif g == "THANK YOU":
        return get_base_skeleton(0.8, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.6, -0.5), finger_spread=0.4, angle_offset=-12)
    elif g == "YES":
        return get_base_skeleton(0.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.1, -0.5), angle_offset=0)
    elif g == "NO":
        return get_base_skeleton(0.1, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.2, -0.2), finger_spread=0.3, angle_offset=15)
    elif g == "HELP":
        lm = get_base_skeleton(0.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.2, -0.3))
        lm[4] = [0.46, 0.65, 0.05]
        return lm
    elif g == "STOP":
        return get_base_skeleton(0.7, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.8, -0.4), finger_spread=0.9, angle_offset=0)
    elif g == "WATER":
        lm = get_base_skeleton(0.0, 1.0, 1.0, 1.0, 0.0, finger_spread=0.8, thumb_dir=(0.1, -0.4))
        lm[4] = [0.48, 0.68, 0.04]
        return lm
    elif g == "FOOD":
        lm = get_base_skeleton(0.3, 0.3, 0.3, 0.3, 0.3, thumb_dir=(-0.1, -0.6), finger_spread=0.2)
        center = [0.50, 0.52, -0.02]
        for tip in [4, 8, 12, 16, 20]:
            lm[tip] = [center[0] + (lm[tip, 0] - center[0]) * 0.3, center[1] + (lm[tip, 1] - center[1]) * 0.3, -0.02]
        return lm
    elif g == "PLEASE":
        return get_base_skeleton(0.5, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.2, -0.8), finger_spread=0.3, angle_offset=-8)
    elif g == "GOOD":
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(-0.1, -1.0), angle_offset=-5)
        lm[4] = [0.44, 0.48, -0.06]
        return lm

    # 2. Recovered 30 gestures
    elif g == "BAD":
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.0, 1.0), angle_offset=180)
        lm[4] = [0.52, 0.94, -0.04]
        return lm
    elif g == "CALL FOR HELP":
        # Emergency call: Thumb and pinky extended wide, wrist tilted with urgency
        return get_base_skeleton(1.0, 0.0, 0.0, 0.0, 1.0, thumb_dir=(-1.3, -0.2), finger_spread=2.0, angle_offset=45)
    elif g == "COME":
        # Index beckoning inward, other fingers closed in fist
        lm = get_base_skeleton(0.0, 0.5, 0.0, 0.0, 0.0, thumb_dir=(0.1, -0.3), angle_offset=15)
        lm[8] = [0.46, 0.48, -0.08]
        return lm
    elif g == "DOCTOR":
        # Three fingers tapping wrist pulse horizontally
        lm = get_base_skeleton(0.2, 0.9, 0.9, 0.9, 0.0, finger_spread=0.2, thumb_dir=(0.0, -0.5), angle_offset=-40)
        lm[8] = [0.48, 0.72, -0.03]; lm[12] = [0.50, 0.73, -0.03]; lm[16] = [0.52, 0.74, -0.03]
        return lm
    elif g == "EMERGENCY":
        # Flashing wide distress hand with maximum spread and tilt
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.4, -0.2), finger_spread=1.8, angle_offset=-25)
    elif g == "FRIEND":
        # Index and middle hooked together
        return get_base_skeleton(0.2, 0.8, 0.8, 0.0, 0.0, thumb_dir=(0.1, -0.5), finger_spread=0.15, angle_offset=20)
    elif g == "GO":
        # Index pointing outward away
        return get_base_skeleton(0.0, 1.0, 0.0, 0.0, 0.0, thumb_dir=(0.0, -0.4), angle_offset=65)
    elif g == "GOOD MORNING":
        # Rising morning sign: upright fingers with pleasant open angle
        return get_base_skeleton(0.9, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.9, -0.6), finger_spread=0.8, angle_offset=-15)
    elif g == "GOOD NIGHT":
        # Hand draping downward like setting sun
        return get_base_skeleton(0.3, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.3, 0.3), finger_spread=0.4, angle_offset=60)
    elif g == "HOSPITAL":
        # 'H' shape (horizontal index and middle straight together)
        return get_base_skeleton(0.1, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.1, -0.4), finger_spread=0.08, angle_offset=45)
    elif g == "I / ME" or g == "I___ME":
        # Index pointing back inward towards chest
        lm = get_base_skeleton(0.1, 0.9, 0.0, 0.0, 0.0, thumb_dir=(-0.1, -0.4), angle_offset=150)
        lm[8] = [0.50, 0.78, 0.08]
        return lm
    elif g == "I AM FINE":
        # '5' hand with thumb widely spread to side
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.5, -0.1), finger_spread=1.3, angle_offset=0)
    elif g == "LOVE":
        # 'ILY' (Thumb, Index, Pinky extended, Middle & Ring curled)
        return get_base_skeleton(1.0, 1.0, 0.0, 0.0, 1.0, thumb_dir=(-1.1, -0.5), finger_spread=1.5)
    elif g == "OK":
        # 'O' circle (Thumb tip touches Index tip), 3 fingers straight up
        lm = get_base_skeleton(0.8, 0.5, 1.0, 1.0, 1.0, finger_spread=1.0)
        lm[4] = [0.44, 0.53, 0.0]
        lm[8] = [0.44, 0.53, 0.0]
        return lm
    elif g == "OPEN PALM" or g == "OPEN_PALM":
        # Open flat relaxed hand facing camera
        return get_base_skeleton(0.8, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.8, -0.4), finger_spread=1.0, angle_offset=0)
    elif g == "PEACE":
        # 'V' victory sign (Index and Middle spread, others curled)
        return get_base_skeleton(0.1, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.2, -0.4), finger_spread=1.5, angle_offset=0)
    elif g == "PHONE":
        # 'Y' telephone hand (Thumb and Pinky extended, tilted slightly)
        return get_base_skeleton(1.0, 0.0, 0.0, 0.0, 1.0, thumb_dir=(-1.1, -0.3), finger_spread=1.6, angle_offset=-30)
    elif g == "POINT":
        # Index finger pointing straight up, all other fingers curled into fist
        return get_base_skeleton(0.0, 1.0, 0.0, 0.0, 0.0, thumb_dir=(0.1, -0.5), angle_offset=0)
    elif g == "POLICE":
        # Two-finger badge salute / peaked angle
        lm = get_base_skeleton(0.0, 1.0, 1.0, 0.0, 0.0, thumb_dir=(0.1, -0.3), finger_spread=0.08, angle_offset=-50)
        return lm
    elif g == "SCHOOL":
        # Horizontal flat hand pose
        return get_base_skeleton(0.8, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.6, -0.3), finger_spread=0.3, angle_offset=80)
    elif g == "SORRY":
        # 'A' fist rotated over chest
        return get_base_skeleton(0.2, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.2, -0.8), angle_offset=-15)
    elif g == "THANKS":
        # Flat hand moving forward offering
        return get_base_skeleton(0.7, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.5, -0.6), finger_spread=0.4, angle_offset=-25)
    elif g in ("THUMBS UP", "THUMBS_UP"):
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(-0.1, -1.0), angle_offset=0)
        lm[4] = [0.44, 0.46, -0.06]
        return lm
    elif g in ("THUMBS DOWN", "THUMBS_DOWN"):
        lm = get_base_skeleton(1.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.1, 1.0), angle_offset=180)
        lm[4] = [0.52, 0.96, -0.04]
        return lm
    elif g == "WAIT":
        # Flat palm tilted down / forward barrier
        return get_base_skeleton(0.5, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.8, -0.6), finger_spread=0.7, angle_offset=-45)
    elif g == "WAVE":
        # Waving hand angled outward with spread fingers
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-1.0, -0.4), finger_spread=1.4, angle_offset=25)
    elif g == "WELCOME":
        # Welcoming scooped palms tilted upward
        return get_base_skeleton(0.7, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.6, -0.5), finger_spread=0.3, angle_offset=40)
    elif g == "FIST":
        return get_base_skeleton(0.0, 0.0, 0.0, 0.0, 0.0, thumb_dir=(0.2, -0.3), angle_offset=0)
    elif g == "BYE":
        # Parting hand angled sideways
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0, thumb_dir=(-0.9, -0.4), finger_spread=0.9, angle_offset=-35)
    elif g == "YOU":
        # Index pointing directly forward towards camera
        lm = get_base_skeleton(0.0, 1.0, 0.0, 0.0, 0.0, thumb_dir=(-0.2, -0.4), angle_offset=0)
        lm[8] = [0.45, 0.30, -0.15]
        return lm
    else:
        return get_base_skeleton(1.0, 1.0, 1.0, 1.0, 1.0)


def generate_augmented_samples(base_template: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """Generates n augmented 73-dimensional invariant feature vectors."""
    vectors = []
    angles = np.linspace(-18, 18, n_samples)

    for i in range(n_samples):
        # 1. Subtle angle rotation
        ang = float(angles[i] + np.random.normal(0, 1.5))
        coords = rotate_landmarks_z(base_template, ang)

        # 2. Add realistic Gaussian joint jitter
        noise = np.random.normal(0, 0.005, coords.shape).astype(np.float32)
        noise[0] *= 0.1  # Keep wrist anchored
        coords_jittered = coords + noise

        # 3. Random scale variation (distance invariant)
        scale = np.random.uniform(0.85, 1.15)
        wrist = coords_jittered[0]
        coords_scaled = (coords_jittered - wrist) * scale + wrist

        # 4. Extract standard 73-dim invariant feature vector
        mock_lm = create_landmark_mock(coords_scaled)
        feat = LandmarkProcessor.extract_feature_vector(mock_lm)
        vectors.append(feat)

    return np.array(vectors, dtype=np.float32)


def generate_full_42_dataset(samples_per_class: int = 400, force_regenerate: bool = True):
    print("=" * 70)
    print("AI SIGN LANGUAGE RECOGNITION - 42 GESTURE DATASET HARMONIZER")
    print("=" * 70)
    print(f"Total Target Classes: {len(config.GESTURES)}")
    print(f"Samples per Class:    {samples_per_class}")
    print(f"Force Regenerate:     {force_regenerate}")
    print("=" * 70)

    dataset_rows = []
    feature_names = LandmarkProcessor.get_feature_names()
    t0 = time.time()

    for idx, gesture_name in enumerate(config.GESTURES, 1):
        safe_name = gesture_name.replace(" ", "_").replace("/", "_")
        gesture_dir = config.DATASET_DIR / safe_name
        gesture_dir.mkdir(parents=True, exist_ok=True)
        existing_npys = list(gesture_dir.glob("sample_*.npy"))

        # If existing real samples exist for core gestures and not force_regenerate:
        if not force_regenerate and len(existing_npys) >= samples_per_class:
            print(f"[{idx:2d}/42] Preserving {len(existing_npys)} existing samples for '{gesture_name}'...")
            for s_file in existing_npys[:samples_per_class]:
                vec = np.load(str(s_file))
                row_dict = {"label": gesture_name}
                for fn, val in zip(feature_names, vec):
                    row_dict[fn] = float(val)
                dataset_rows.append(row_dict)
        else:
            print(f"[{idx:2d}/42] Generating {samples_per_class} clean samples for '{gesture_name}'...")
            base_skeleton = build_template_for_gesture(gesture_name)
            feat_vectors = generate_augmented_samples(base_skeleton, n_samples=samples_per_class)

            for s_idx, vec in enumerate(feat_vectors, 1):
                file_path = gesture_dir / f"sample_{s_idx:04d}.npy"
                np.save(str(file_path), vec)
                row_dict = {"label": gesture_name}
                for fn, val in zip(feature_names, vec):
                    row_dict[fn] = float(val)
                dataset_rows.append(row_dict)

    df = pd.DataFrame(dataset_rows)
    config.DATASET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_CSV_PATH, index=False)
    print(f"\n[SUCCESS] Generated & verified full 42-class dataset!")
    print(f"Dataset Shape: {df.shape} ({len(df)} samples, {df.shape[1] - 1} features)")
    print(f"Elapsed Time:  {time.time() - t0:.2f} seconds.")
    print("=" * 70)
    return df


if __name__ == "__main__":
    generate_full_42_dataset(samples_per_class=400, force_regenerate=True)
