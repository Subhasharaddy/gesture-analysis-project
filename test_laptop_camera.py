"""
Comprehensive Camera Diagnostic & Verification Tool.
Tests both Laptop Webcam and Android Phone Camera (USB / Stream).
"""

import cv2
import sys
import shutil
import subprocess
import urllib.request
from typing import Optional, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def check_stream_url(url: str) -> bool:
    """Quick check for local stream availability."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenCV-Client"})
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_comprehensive_diagnostic():
    print("=" * 60)
    print("      AI SIGN RECOGNITION - CAMERA DIAGNOSTIC SYSTEM")
    print("=" * 60)

    # 1. LAPTOP WEBCAM TEST
    print("\n[1] TESTING LAPTOP BUILT-IN WEBCAM (Index 0)...")
    cap_laptop = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    laptop_working = False
    laptop_res = "N/A"

    if cap_laptop.isOpened():
        ret, frame = cap_laptop.read()
        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            laptop_res = f"{w} x {h}"
            laptop_working = True
        cap_laptop.release()

    if laptop_working:
        print(f"  ✓ STATUS: WORKING & STREAMING PROPERLY")
        print(f"  ✓ BACKEND: DirectShow (CAP_DSHOW)")
        print(f"  ✓ NATIVE RESOLUTION: {laptop_res}")
    else:
        print("  ✗ STATUS: NOT WORKING / NOT ACCESSIBLE")
        print("  Troubleshooting:")
        print("    1. Windows Settings -> Privacy & Security -> Camera -> Enable Access")
        print("    2. Check privacy shutter or keyboard Fn key (F8 / F10 / F6)")
        print("    3. Ensure no other application (Teams, Zoom, Browser) is using the webcam.")

    # 2. ANDROID PHONE USB CAMERA TEST (Indices 1 to 3)
    print("\n[2] TESTING ANDROID PHONE USB CAMERA (Indices 1 to 3)...")
    phone_found = False
    phone_info = "None detected"

    for idx in [1, 2, 3]:
        cap_ext = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap_ext.isOpened():
            ret, frame = cap_ext.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                phone_found = True
                phone_info = f"Index {idx} ({w} x {h})"
                cap_ext.release()
                break
            cap_ext.release()

    if phone_found:
        print(f"  ✓ STATUS: PHONE CAMERA CONNECTED VIA USB")
        print(f"  ✓ DETAILS: {phone_info}")
    else:
        print("  ✗ STATUS: NO USB WEBCAM FEED DETECTED ON INDICES 1-3")

    # 3. ANDROID PHONE STREAM / BRIDGE TEST
    print("\n[3] TESTING ANDROID STREAM & ADB BRIDGES...")
    adb_path = shutil.which("adb")
    adb_devices = []
    if adb_path:
        try:
            res = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.startswith("List of")]
            adb_devices = [l.split()[0] for l in lines if "\tdevice" in l]
        except Exception:
            pass

    if adb_devices:
        print(f"  ✓ ADB CONNECTED DEVICES: {adb_devices}")
    else:
        print("  - ADB: No device bridged or ADB not in PATH")

    droidcam_stream = check_stream_url("http://127.0.0.1:4747/video")
    ipwebcam_stream = check_stream_url("http://127.0.0.1:8080/video")
    print(f"  - DROIDCAM USB STREAM (127.0.0.1:4747): {'ACTIVE' if droidcam_stream else 'INACTIVE'}")
    print(f"  - IP WEBCAM STREAM (127.0.0.1:8080): {'ACTIVE' if ipwebcam_stream else 'INACTIVE'}")

    # 4. SUMMARY & RECOMMENDATIONS
    print("\n" + "=" * 60)
    print("                    DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if laptop_working:
        print("✓ LAPTOP WEBCAM: READY (Primary camera working properly)")
    else:
        print("✗ LAPTOP WEBCAM: FAILED TO CAPTURE FRAMES")

    if phone_found or droidcam_stream or ipwebcam_stream:
        print("✓ PHONE CAMERA: READY")
    else:
        print("ℹ PHONE CAMERA: NOT STREAMING YET")
        print("  To use your Android phone as a camera:")
        print("  Option A (Android 14+): Connect USB cable -> Tap notification -> Select 'Webcam'")
        print("  Option B: Install DroidCam app on Phone & PC -> Start USB or Wi-Fi streaming")
        print("  Option C: Install IP Webcam app on Phone -> Start Server -> Click 'Phone Stream URL' in UI")

    print("=" * 60)


if __name__ == "__main__":
    run_comprehensive_diagnostic()
