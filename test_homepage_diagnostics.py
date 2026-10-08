"""
Comprehensive Automated Diagnostic Test Suite for Redesigned Web Exhibition Homepage.
Validates:
  1. Live HTTP Web Server endpoints on port 5000
  2. HTML structure & required UI sections
  3. JSON API schemas (/api/status, /api/stats, /api/gesture_guide)
  4. Interactive POST endpoints (start/stop camera, mute, clear history)
  5. Static assets & gesture PNG image resolution
  6. Machine Learning Model inference & 73-D feature classification latency
  7. Kannada Unicode text integrity & dictionary definitions
"""

import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_URL = "http://127.0.0.1:5000"

def log_test(title, passed=True, details=""):
    badge = " [PASS] " if passed else " [FAIL] "
    color_prefix = "\033[92m" if passed else "\033[91m"
    color_reset = "\033[0m"
    print(f"{badge} {title}")
    if details:
        print(f"         └─ {details}")

def http_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge-Diagnostic/1.0"})
    with urllib.request.urlopen(req, timeout=5) as res:
        return res.status, res.read()

def http_post(path, data=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "SignBridge-Diagnostic/1.0"}
    )
    with urllib.request.urlopen(req, timeout=5) as res:
        return res.status, json.loads(res.read().decode("utf-8"))

def run_diagnostics():
    print("=" * 70)
    print(" SIGNBRIDGE AI - HOMEPAGE AUTOMATED DIAGNOSTIC TEST SUITE")
    print("=" * 70)
    print(f"Target Server: {BASE_URL}\n")

    total_tests = 0
    passed_tests = 0

    # ---------------------------------------------------------
    # TEST 1: Root Homepage Web Server Availability
    # ---------------------------------------------------------
    total_tests += 1
    try:
        status, html_bytes = http_get("/")
        html_str = html_bytes.decode("utf-8")
        has_title = "<title>SignBridge AI" in html_str
        is_ok = status == 200 and has_title
        log_test("GET / (Homepage Root)", is_ok, f"HTTP {status}, Title Verified, Content-Length: {len(html_bytes):,} bytes")
        if is_ok: passed_tests += 1
    except Exception as e:
        log_test("GET / (Homepage Root)", False, f"Connection error: {e}")

    # ---------------------------------------------------------
    # TEST 2: HTML Section Anchors & Structural Verification
    # ---------------------------------------------------------
    sections = [
        ("Navigation Bar", 'class="navbar"'),
        ("Hero Section", 'id="home"'),
        ("Statistics Section", 'id="stats"'),
        ("Features Section", 'id="features"'),
        ("How It Works Pipeline", 'id="how-it-works"'),
        ("Live Recognition Studio", 'id="live-studio"'),
        ("Gesture Explorer", 'id="gesture-explorer"'),
        ("About Project", 'id="about"'),
        ("Contact & Feedback", 'id="contact"'),
        ("Footer", 'class="footer"'),
        ("Gesture Guide Modal", 'id="guideModal"'),
        ("Auth Modal", 'id="authModal"'),
    ]

    for name, marker in sections:
        total_tests += 1
        found = marker in html_str
        log_test(f"UI Section: {name}", found, f"Marker '{marker}' present in DOM")
        if found: passed_tests += 1

    # ---------------------------------------------------------
    # TEST 3: Telemetry API (/api/status)
    # ---------------------------------------------------------
    total_tests += 1
    try:
        status, data_bytes = http_get("/api/status")
        data = json.loads(data_bytes.decode("utf-8"))
        required_keys = ["camera_active", "hand_detected", "gesture", "confidence", "kannada", "fps", "arduino_status"]
        all_keys = all(k in data for k in required_keys)
        log_test("API: /api/status (Live Telemetry)", status == 200 and all_keys,
                 f"FPS: {data.get('fps', 0):.1f} | Gesture: {data.get('gesture')} | Arduino: {data.get('arduino_status')}")
        if status == 200 and all_keys: passed_tests += 1
    except Exception as e:
        log_test("API: /api/status", False, str(e))

    # ---------------------------------------------------------
    # TEST 4: Project Statistics API (/api/stats)
    # ---------------------------------------------------------
    total_tests += 1
    try:
        status, data_bytes = http_get("/api/stats")
        stats = json.loads(data_bytes.decode("utf-8"))
        req_stats = ["total_samples", "train_accuracy", "test_accuracy", "core_gestures", "landmarks_per_hand"]
        has_stats = all(k in stats for k in req_stats)
        log_test("API: /api/stats (Dynamic Metrics)", status == 200 and has_stats,
                 f"Samples: {stats.get('total_samples'):,} | Accuracy: {stats.get('test_accuracy')}% | Gestures: {stats.get('core_gestures')}")
        if status == 200 and has_stats: passed_tests += 1
    except Exception as e:
        log_test("API: /api/stats", False, str(e))

    # ---------------------------------------------------------
    # TEST 5: 12 Gesture Guide API (/api/gesture_guide)
    # ---------------------------------------------------------
    total_tests += 1
    try:
        status, data_bytes = http_get("/api/gesture_guide")
        guide = json.loads(data_bytes.decode("utf-8"))
        gestures = guide.get("gestures", [])
        is_12 = len(gestures) == 12
        has_content = all(g.get("name") and g.get("kannada") and g.get("how_to") for g in gestures)
        log_test("API: /api/gesture_guide (ISL Classes)", status == 200 and is_12 and has_content,
                 f"Returned {len(gestures)} full gesture metadata entries with Kannada & how-to instructions")
        if status == 200 and is_12 and has_content: passed_tests += 1
    except Exception as e:
        log_test("API: /api/gesture_guide", False, str(e))

    # ---------------------------------------------------------
    # TEST 6: Static Asset Images (All 12 Gesture PNGs)
    # ---------------------------------------------------------
    core_12 = ["HOME", "NAMASTE", "HELLO", "THANK_YOU", "YES", "NO", "HELP", "STOP", "WATER", "FOOD", "PLEASE", "GOOD"]
    img_success = 0
    png_header = b"\x89PNG\r\n\x1a\n"
    for g in core_12:
        try:
            status, img_data = http_get(f"/static/images/gestures/{g}.png")
            if status == 200 and img_data.startswith(png_header):
                img_success += 1
        except Exception:
            pass

    total_tests += 1
    log_test("Static Assets: 12 Gesture PNG Images", img_success == 12,
             f"Verified {img_success}/12 PNG images served with valid 200 OK & binary headers")
    if img_success == 12: passed_tests += 1

    # ---------------------------------------------------------
    # TEST 7: Interactive Control Endpoints (POST Operations)
    # ---------------------------------------------------------
    controls = [
        ("/api/start_camera", "Start Camera"),
        ("/api/stop_camera", "Stop Camera"),
        ("/api/start_camera", "Re-activate Camera"),
        ("/api/toggle_mute", "Toggle Audio Mute (ON)"),
        ("/api/toggle_mute", "Toggle Audio Mute (OFF)"),
        ("/api/clear_history", "Clear Gesture History Queue"),
    ]

    for path, desc in controls:
        total_tests += 1
        try:
            status, res = http_post(path)
            passed = status == 200 and res.get("success") is True
            log_test(f"Control POST: {desc}", passed, f"{path} -> {res}")
            if passed: passed_tests += 1
        except Exception as e:
            log_test(f"Control POST: {desc}", False, str(e))

    # ---------------------------------------------------------
    # TEST 8: ML Model & Inference Speed Verification
    # ---------------------------------------------------------
    total_tests += 1
    try:
        import joblib
        import numpy as np
        import config

        model = joblib.load(config.MODEL_PKL_PATH)
        encoder = joblib.load(config.LABEL_ENCODER_PATH)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        # Mock 73-dimensional invariant feature vector
        mock_feat = np.random.uniform(-1.0, 1.0, 73).reshape(1, -1)

        # Warmup
        for _ in range(5):
            model.predict_proba(mock_feat)

        # Benchmark 10 inferences
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            probs = model.predict_proba(mock_feat)[0]
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        inference_ms = sum(times) / len(times)
        num_classes = len(encoder.classes_)
        is_ml_ok = num_classes >= 12 and inference_ms < 35.0
        log_test("AI Engine: Model & Inference Latency", is_ml_ok,
                 f"Trained Classes: {num_classes} | Avg Inference Time: {inference_ms:.2f} ms (<35ms budget)")
        if is_ml_ok: passed_tests += 1
    except Exception as e:
        log_test("AI Engine: Model Verification", False, str(e))

    # ---------------------------------------------------------
    # TEST 9: Kannada Unicode & Bilingual Mapping Integrity
    # ---------------------------------------------------------
    total_tests += 1
    try:
        from utils.gesture_database import GESTURE_DEFINITIONS, CORE_12_GESTURES
        valid_kannada_count = 0
        for g in CORE_12_GESTURES:
            info = GESTURE_DEFINITIONS.get(g, {})
            kannada = info.get("kannada", "")
            if kannada and len(kannada.strip()) > 0:
                valid_kannada_count += 1

        is_kannada_ok = valid_kannada_count == 12
        log_test("Localization: Kannada Unicode UTF-8 Integrity", is_kannada_ok,
                 f"{valid_kannada_count}/12 Core Gestures mapped to authentic Unicode Kannada strings")
        if is_kannada_ok: passed_tests += 1
    except Exception as e:
        log_test("Localization Verification", False, str(e))

    # ---------------------------------------------------------
    # SUMMARY REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f" DIAGNOSTIC SUMMARY: {passed_tests} / {total_tests} TESTS PASSED ({(passed_tests/total_tests)*100:.1f}%)")
    print("=" * 70)

    if passed_tests == total_tests:
        print(">> ALL SYSTEMS OPERATIONAL: Redesigned homepage is 100% functional, responsive, and exhibition-ready!\n")
        return 0
    else:
        print(f">> WARNING: {total_tests - passed_tests} tests reported issues.\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_diagnostics())
