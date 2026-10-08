"""
Visual Reference Generator for 12 Sign Language Gestures.
Creates exhibition-quality visual reference guide diagrams for each of the
12 gesture classes in `images/gestures/<GESTURE>.png`.

Visual elements:
- Dark exhibition card theme
- Skeletal 21-joint landmark diagram rendered with glowing joints & bone connections
- ISL category badge
- English title & Kannada translation
- Step-by-step instructions for performing the sign in front of the camera
"""

import math
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import config
from utils.gesture_database import GESTURE_DEFINITIONS
from utils.hand_tracker import HAND_CONNECTIONS
from generate_42_gestures_dataset import build_template_for_gesture

ALL_GESTURES = config.ALL_GESTURES


def draw_glow_circle(img_bgr, center, radius, color, glow_color, glow_radius=10):
    """Draws a circle with smooth outer neon glow."""
    cx, cy = center
    for r in range(glow_radius, radius, -2):
        alpha = 0.15 * (1.0 - (r - radius) / (glow_radius - radius + 1e-5))
        overlay = img_bgr.copy()
        cv2.circle(overlay, (cx, cy), r, glow_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0, img_bgr)
    cv2.circle(img_bgr, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img_bgr, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)


def generate_guide_image(gesture_name: str, output_path: Path):
    """Renders a 600x600 high-resolution visual reference guide card."""
    card_w, card_h = 600, 600
    # Dark modern canvas
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (18, 22, 28)  # Deep dark navy-slate

    # Subtle radial gradient center
    for r in range(280, 0, -20):
        intensity = int(12 * (1.0 - r / 280.0))
        cv2.circle(card, (card_w // 2, 280), r, (18 + intensity, 22 + intensity, 32 + intensity), -1)

    # Card border with glowing corner accents
    cv2.rectangle(card, (12, 12), (card_w - 12, card_h - 12), (48, 56, 70), 2)
    accent_len = 28
    corner_color = (0, 210, 255)
    # Corners
    cv2.line(card, (12, 12), (12 + accent_len, 12), corner_color, 3)
    cv2.line(card, (12, 12), (12, 12 + accent_len), corner_color, 3)
    cv2.line(card, (card_w - 12, 12), (card_w - 12 - accent_len, 12), corner_color, 3)
    cv2.line(card, (card_w - 12, 12), (card_w - 12, 12 + accent_len), corner_color, 3)
    cv2.line(card, (12, card_h - 12), (12 + accent_len, card_h - 12), corner_color, 3)
    cv2.line(card, (12, card_h - 12), (12, card_h - 12 - accent_len), corner_color, 3)
    cv2.line(card, (card_w - 12, card_h - 12), (card_w - 12 - accent_len, card_h - 12), corner_color, 3)
    cv2.line(card, (card_w - 12, card_h - 12), (card_w - 12, card_h - 12 - accent_len), corner_color, 3)

    # Header Box
    cv2.rectangle(card, (20, 20), (card_w - 20, 95), (26, 32, 42), -1)
    cv2.line(card, (20, 95), (card_w - 20, 95), (60, 70, 85), 1)

    info = GESTURE_DEFINITIONS.get(gesture_name, {})
    is_emg = info.get("is_emergency", False)

    # Tag Badge: OFFICIAL ISL SIGN or EMERGENCY
    badge_color = (40, 40, 230) if is_emg else (46, 160, 67)
    badge_text = "EMERGENCY ALERT SIGN" if is_emg else "OFFICIAL ISL SIGN"
    cv2.rectangle(card, (card_w - 200, 32), (card_w - 32, 54), badge_color, -1)
    cv2.putText(card, badge_text, (card_w - 192, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Gesture Name
    cv2.putText(card, gesture_name, (36, 62), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

    # Subtitle English Meaning
    meaning_text = info.get("english", gesture_name)
    cv2.putText(card, f"Meaning: {meaning_text}", (36, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

    # Render Skeletal Hand Diagram
    base_lm = build_template_for_gesture(gesture_name)
    # Map normalized coordinates (around 0.5, 0.6) to pixel coordinates in (70 to 530, 110 to 440)
    xs = base_lm[:, 0]
    ys = base_lm[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    box_cx = card_w // 2
    box_cy = 275
    box_scale = min(280.0 / max(0.25, max_x - min_x), 260.0 / max(0.25, max_y - min_y))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    pts = []
    for lm in base_lm:
        px = int(box_cx + (lm[0] - center_x) * box_scale)
        py = int(box_cy + (lm[1] - center_y) * box_scale)
        pts.append((px, py))

    # Draw Bone Connections (Glowing cyan/blue)
    bone_color = (235, 185, 0) if not is_emg else (60, 60, 255)
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        pt1 = pts[p1_idx]
        pt2 = pts[p2_idx]
        cv2.line(card, pt1, pt2, (bone_color[0] // 3, bone_color[1] // 3, bone_color[2] // 3), 4, cv2.LINE_AA)
        cv2.line(card, pt1, pt2, bone_color, 2, cv2.LINE_AA)

    # Draw Joint Circles
    for i, pt in enumerate(pts):
        is_tip = (i in [4, 8, 12, 16, 20])
        is_wrist = (i == 0)
        if is_tip:
            draw_glow_circle(card, pt, radius=7, color=(0, 255, 120), glow_color=(0, 210, 80), glow_radius=14)
        elif is_wrist:
            draw_glow_circle(card, pt, radius=6, color=(255, 200, 0), glow_color=(255, 180, 0), glow_radius=12)
        else:
            cv2.circle(card, pt, 4, (240, 240, 240), -1, cv2.LINE_AA)
            cv2.circle(card, pt, 4, (30, 36, 46), 1, cv2.LINE_AA)

    # Landmark labels for key tips
    tip_labels = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
    for tip_idx, tip_name in tip_labels.items():
        tx, ty = pts[tip_idx]
        offset_y = -12 if ty < box_cy else 18
        cv2.putText(card, tip_name, (tx - 16, ty + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 200, 220), 1, cv2.LINE_AA)

    # Instructions Bottom Card
    cv2.rectangle(card, (20, 460), (card_w - 20, card_h - 20), (24, 30, 40), -1)
    cv2.line(card, (20, 460), (card_w - 20, 460), (55, 65, 80), 1)

    cv2.putText(card, "HOW TO PERFORM:", (32, 484), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 210, 255), 1, cv2.LINE_AA)

    how_text = info.get("how_to_perform", info.get("description", "Hold gesture steady in front of camera."))
    # Word wrap instruction text
    words = how_text.split()
    line1, line2 = "", ""
    for w in words:
        if len(line1) + len(w) + 1 < 46:
            line1 += (w + " ")
        else:
            line2 += (w + " ")

    cv2.putText(card, f"• {line1.strip()}", (32, 508), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 237, 243), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(card, f"  {line2.strip()}", (32, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 237, 243), 1, cv2.LINE_AA)

    kannada_translit = info.get("kannada_translit", "")
    kannada_text = info.get("kannada", "")
    speech_phrase = info.get("speech", gesture_name)
    cv2.putText(card, f"Kannada: {kannada_translit}  |  Speech: \"{speech_phrase}\"", (32, 556), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 150, 165), 1, cv2.LINE_AA)
    cv2.putText(card, "Exhibition Tip: Hold posture steady at 1-2 feet distance from laptop webcam.", (32, 574), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 180, 120), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), card)


def generate_all_guide_images():
    """Generates guide reference cards for all 42 gesture classes."""
    images_dir = config.IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    static_images_dir = config.BASE_DIR / "static" / "images" / "gestures"
    static_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visual reference guides for {len(ALL_GESTURES)} gestures...")
    for idx, g in enumerate(ALL_GESTURES, 1):
        safe_name = g.replace(" ", "_").replace("/", "_")
        target_path = images_dir / f"{safe_name}.png"
        generate_guide_image(g, target_path)

        # Also copy to web static folder for dashboard access
        static_path = static_images_dir / f"{safe_name}.png"
        import shutil
        shutil.copyfile(str(target_path), str(static_path))

        print(f"  [{idx:2d}/42] Generated: {target_path.name}")

    print(f"[SUCCESS] All 42 gesture guide cards generated in {images_dir}")


if __name__ == "__main__":
    generate_all_guide_images()
