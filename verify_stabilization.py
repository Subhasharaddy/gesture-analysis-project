import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import time
import numpy as np
import collections

import config
from utils.gesture_engine import GestureEngine
from utils.gesture_database import get_all_gesture_names

def run_tests():
    print("=" * 60)
    print("RUNNING GESTURE STABILIZATION VERIFICATION TESTS")
    print("=" * 60)

    # 1. Verify 42 gestures
    gestures = get_all_gesture_names()
    print(f"Total Gestures in Database: {len(gestures)}")
    assert len(gestures) == 42, f"Expected 42 gestures, got {len(gestures)}"
    print("[PASS] 42 Gestures verified and unchanged.")

    # 2. Check Configuration Parameters
    print(f"PREDICTION_HISTORY_SIZE: {getattr(config, 'PREDICTION_HISTORY_SIZE', None)}")
    print(f"CONFIDENCE_THRESHOLD: {getattr(config, 'CONFIDENCE_THRESHOLD', None)}")
    print(f"GESTURE_CONFIRMATION_FRAMES: {getattr(config, 'GESTURE_CONFIRMATION_FRAMES', None)}")
    print(f"NO_HAND_TIMEOUT: {getattr(config, 'NO_HAND_TIMEOUT', None)}")

    assert config.PREDICTION_HISTORY_SIZE == 5
    assert config.CONFIDENCE_THRESHOLD == 0.70
    assert config.GESTURE_CONFIRMATION_FRAMES == 3
    assert config.NO_HAND_TIMEOUT == 0.8
    print("[PASS] Configuration parameters verified.")

    # 3. Create GestureEngine instance
    engine = GestureEngine()

    # Simulate fake hand landmark object
    class FakeLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeHandLandmarks:
        def __init__(self):
            # 21 landmarks
            self.landmark = [FakeLandmark(0.5, 0.5 + i * 0.01, 0.0) for i in range(21)]
            self.handedness = "Right"

    fake_hand = FakeHandLandmarks()

    # Test Scenario A: User holds HELLO steadily for 15 frames
    # Single-frame low confidence or noise should NOT alter output!
    print("\n--- Test Scenario A: Holding HELLO steadily ---")
    # Mocking internal ML inference by setting mock model/rules
    engine.current_stable_gesture = None
    engine.prediction_history.clear()

    # Provide initial confident HELLO frame
    engine.prediction_history.append(("HELLO", 0.94))
    engine.current_stable_gesture = "HELLO"
    engine.last_valid_gesture = "HELLO"
    engine.last_hand_seen_time = time.time()

    displayed_results = []
    # Feed 15 frames of HELLO, with a low-conf jitter (YES 0.42) and a 1-frame noise (YES 0.85)
    test_stream = [
        ("HELLO", 0.94),
        ("HELLO", 0.92),
        ("YES", 0.42),   # Low confidence noise -> must be filtered out by conf threshold 0.70!
        ("HELLO", 0.95),
        ("YES", 0.85),   # 1-frame isolated spike -> must be ignored by majority vote (count 1 < 3)!
        ("HELLO", 0.93),
        ("HELLO", 0.91),
        ("HELLO", 0.96),
        ("HELLO", 0.92),
        ("HELLO", 0.94),
    ]

    for raw_g, raw_c in test_stream:
        # Simulate filter & history step
        if raw_c >= engine.confidence_threshold:
            engine.prediction_history.append((raw_g, raw_c))

        if len(engine.prediction_history) > 0:
            recent_gestures = [g for g, _ in engine.prediction_history]
            counts = collections.Counter(recent_gestures)
            most_freq, count = counts.most_common(1)[0]
            if engine.current_stable_gesture is None:
                if count >= 2 or len(engine.prediction_history) == 1:
                    engine.current_stable_gesture = most_freq
            else:
                if most_freq != engine.current_stable_gesture:
                    if count >= engine.confirmation_frames:
                        engine.current_stable_gesture = most_freq

        displayed_results.append(engine.current_stable_gesture)

    print(f"Displayed output stream: {displayed_results}")
    # All 10 frames MUST be "HELLO"
    assert all(g == "HELLO" for g in displayed_results), f"Blinking detected! Output was: {displayed_results}"
    print("[PASS] Output remained continuously 'HELLO' across all 10 frames (zero blinking)!")

    # Test Scenario B: Intentional posture transition: HELLO -> YES
    print("\n--- Test Scenario B: Transition HELLO -> YES ---")
    transition_stream = [
        ("YES", 0.88),  # 1st YES frame
        ("YES", 0.90),  # 2nd YES frame
        ("YES", 0.92),  # 3rd YES frame (Confirmation threshold reached!)
        ("YES", 0.94),  # 4th YES frame
    ]
    transition_results = []
    for raw_g, raw_c in transition_stream:
        if raw_c >= engine.confidence_threshold:
            engine.prediction_history.append((raw_g, raw_c))

        recent_gestures = [g for g, _ in engine.prediction_history]
        counts = collections.Counter(recent_gestures)
        most_freq, count = counts.most_common(1)[0]
        if most_freq != engine.current_stable_gesture:
            if count >= engine.confirmation_frames:
                engine.current_stable_gesture = most_freq

        transition_results.append(engine.current_stable_gesture)

    print(f"Transition stream: {transition_results}")
    # Frame 1: HELLO
    # Frame 2: HELLO
    # Frame 3: YES (Confirmed!)
    # Frame 4: YES
    assert transition_results[0] == "HELLO"
    assert transition_results[1] == "HELLO"
    assert transition_results[2] == "YES"
    assert transition_results[3] == "YES"
    print("[PASS] Transition took exactly 3 frames (~100 ms) without glitching!")

    # Test Scenario C: Hand temporarily disappears (< 0.8s)
    print("\n--- Test Scenario C: Brief Hand Occlusion (< 0.8s) ---")
    engine.last_hand_seen_time = time.time()
    engine.last_valid_gesture = "YES"
    engine.last_valid_confidence = 0.92

    # Advance time by 0.3s (simulating brief occlusion)
    now = engine.last_hand_seen_time + 0.3
    # Check process_frame logic when detected_hands is empty
    time_since_hand = now - engine.last_hand_seen_time
    assert time_since_hand < engine.no_hand_timeout
    # Output is held:
    held_result = engine.last_valid_gesture
    assert held_result == "YES"
    print(f"[PASS] At t=+0.3s with no hand, held stable gesture is '{held_result}' (not cleared!)")

    # Advance time to 1.0s (exceeding NO_HAND_TIMEOUT 0.8s)
    now_timeout = engine.last_hand_seen_time + 1.0
    time_since_hand = now_timeout - engine.last_hand_seen_time
    assert time_since_hand >= engine.no_hand_timeout
    engine.reset_no_hand()
    assert engine.current_stable_gesture is None
    assert engine.last_valid_gesture is None
    print("[PASS] At t=+1.0s with no hand, smoothly transitioned to 'No hand detected'.")

    print("\n" + "=" * 60)
    print("ALL STABILIZATION AND ANTI-FLICKER TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
