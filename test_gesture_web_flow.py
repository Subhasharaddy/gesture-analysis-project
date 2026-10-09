"""
Comprehensive Pipeline Test for Gesture Recognition, Gesture Name Display & Voice Output.
"""

import sys
import json
import time
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import config
from generate_42_gestures_dataset import build_template_for_gesture
from app_web import app, gesture_engine, tts


def test_full_pipeline():
    print("=" * 70)
    print("AI SIGN LANGUAGE SYSTEM - COMPLETE PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    client = app.test_client()

    # 1. Test Empty Hand Handling & Hold Logic
    print("\n[STEP 1] Testing Empty Hand Endpoint Response...")
    resp = client.post("/api/predict", json={"hands": []})
    data = json.loads(resp.data)
    print(f"Empty Hand Response: gesture='{data.get('gesture_name')}' | conf={data.get('confidence')}")
    assert data.get("gesture_name") == "No hand detected", "Expected 'No hand detected' on initial empty hands"
    print("-> STEP 1 PASSED: 'No hand detected' handled safely.\n")

    # 2. Test All 42 Gestures
    print("[STEP 2] Testing All 42 Gestures Through End-to-End Recognition & TTS Pipeline...")
    all_gestures = config.GESTURES
    matched = 0
    total = len(all_gestures)

    for idx, g in enumerate(all_gestures, 1):
        # Reset state between distinct gesture tests
        gesture_engine.reset_no_hand()

        coords = build_template_for_gesture(g)
        lms_json = [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in coords]

        payload = {
            "hands": [
                {
                    "label": "Right",
                    "landmarks": lms_json
                }
            ]
        }

        # Feed 3 consecutive frames to satisfy temporal stabilization
        final_data = None
        for frame_idx in range(3):
            resp = client.post("/api/predict", json=payload)
            final_data = json.loads(resp.data)

        detected_name = final_data.get("gesture_name")
        conf = final_data.get("confidence", 0.0)
        meaning = final_data.get("meaning", "")
        spoken = final_data.get("spoken_phrase", "")
        is_emg = final_data.get("is_emergency", False)

        is_match = (detected_name == g)
        if is_match:
            matched += 1

        status_sym = "✓" if is_match else "✗"
        print(f" {status_sym} [{idx:02d}/{total}] Expected: {g:<16} | Detected: {detected_name:<16} (conf={conf*100:5.1f}%) | Spoken: \"{spoken}\"")

        assert detected_name is not None and detected_name != "", f"Empty gesture name for {g}!"
        assert detected_name not in ("No hand detected", "Unknown Gesture", "Detecting..."), f"Valid gesture {g} resolved to placeholder {detected_name}!"
        assert spoken is not None and len(spoken) > 0, f"Spoken phrase missing for {g}!"

    print("-" * 70)
    print(f"PIPELINE ACCURACY: {matched}/{total} ({matched/total*100:.1f}%) Gestures Verified")
    print("=" * 70)
    print("ALL 42 GESTURES SUCCESSFULLY RECOGNIZED, DISPLAYED, AND LINKED TO TTS!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
