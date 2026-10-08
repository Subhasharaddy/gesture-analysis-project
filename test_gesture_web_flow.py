import json
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import config
from app_web import app, gesture_engine

def test_flow():
    print("========================================")
    print("TESTING GESTURE PREDICT & WEB FLOW")
    print("========================================")
    
    client = app.test_client()
    
    # Test 1: Empty hand payload returns "No hand detected"
    resp = client.post("/api/predict", json={"hands": []})
    data = json.loads(resp.data)
    print(f"Test 1 (Empty hands) Response: {data}")
    assert data.get("gesture") == "No hand detected", f"Expected 'No hand detected', got {data.get('gesture')}"
    assert data.get("gesture_name") == "No hand detected", f"Expected 'No hand detected', got {data.get('gesture_name')}"
    print("-> Test 1 PASSED: 'No hand detected' returned correctly when no hand available.\n")
    
    # Test 2: Recognized gesture test with sample data
    # Find any available sample in dataset
    sample_gesture = "HELLO"
    sample_files = list((config.DATASET_DIR / sample_gesture).glob("sample_*.npy"))
    if not sample_files:
        # Try any folder
        for d in config.DATASET_DIR.iterdir():
            if d.is_dir():
                files = list(d.glob("sample_*.npy"))
                if files:
                    sample_gesture = d.name
                    sample_files = files
                    break
                    
    assert sample_files, "No dataset samples found to test!"
    print(f"Using sample for gesture: {sample_gesture} ({sample_files[0].name})")
    
    feat = np.load(str(sample_files[0]))
    raw_63 = feat[:63].reshape(21, 3)
    lms_json = [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in raw_63]
    
    payload = {
        "hands": [
            {
                "label": "Right",
                "landmarks": lms_json
            }
        ]
    }
    
    # Reset engine state first
    gesture_engine.reset_no_hand()
    
    # Feed 2-3 frames to ensure stability window confirms gesture
    for i in range(3):
        resp = client.post("/api/predict", json=payload)
        data = json.loads(resp.data)
        print(f"Frame {i+1} Response -> gesture: '{data.get('gesture')}', gesture_name: '{data.get('gesture_name')}', conf: {data.get('confidence')*100:.1f}%, new_conf: {data.get('is_new_confirmation')}")
        
    final_gesture_name = data.get("gesture_name")
    assert final_gesture_name is not None and len(final_gesture_name) > 0, "gesture_name is missing or empty!"
    assert final_gesture_name != "No hand detected", f"Expected recognized gesture, got {final_gesture_name}"
    assert data.get("gesture") == final_gesture_name, "gesture and gesture_name must match"
    print(f"\n-> Test 2 PASSED: Recognized gesture '{final_gesture_name}' successfully returned by backend!")
    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    test_flow()
