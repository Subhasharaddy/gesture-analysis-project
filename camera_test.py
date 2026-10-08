"""
Camera Diagnostic & Test Tool for AI Sign Language Recognition.
Scans camera indexes 0 through 5, identifies available devices,
displays real-time preview, and allows seamless switching between
Laptop Built-in Webcam and Android Phone Camera.
"""

import cv2
import time
import sys
from typing import List, Tuple

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import config
from utils.camera_manager import CameraManager


def scan_all_cameras(max_tested: int = 6) -> List[Tuple[int, str, bool, int, int]]:
    """
    Scans camera indexes 0 through 5.
    Returns list of tuples: (index, source_type, is_available, width, height)
    """
    print("=" * 45)
    print("CAMERA DETECTION")
    print("=" * 45)

    results = []
    for idx in range(max_tested):
        # DirectShow on Windows avoids freeze
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        is_avail = cap.isOpened()
        w, h = 0, 0
        if is_avail:
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                is_avail = False
            cap.release()

        # Label source type
        if idx == 0:
            source_type = "Laptop Webcam"
        else:
            source_type = "Android Phone Camera"

        if is_avail:
            print(f"Camera {idx} : AVAILABLE ({w}x{h}) - {source_type}")
        else:
            print(f"Camera {idx} : NOT AVAILABLE")

        results.append((idx, source_type, is_avail, w, h))

    return results


def run_camera_test(preferred_idx: int = None):
    scan_results = scan_all_cameras(max_tested=6)

    available_cams = [r for r in scan_results if r[2]]

    if not available_cams:
        print("\n" + "=" * 45)
        print("[ERROR] No working cameras detected on this PC!")
        print("Troubleshooting Tips:")
        print("  1. Windows Settings -> Privacy & security -> Camera -> Enable 'Let desktop apps access camera'")
        print("  2. Ensure no other application (Zoom, Teams, Skype, Browser) is using the camera.")
        print("  3. For Android Phone: Use a USB DATA cable and enable Webcam mode or DroidCam.")
        print("=" * 45)
        return

    # Determine default active index
    if preferred_idx is not None and any(r[0] == preferred_idx and r[2] for r in scan_results):
        active_idx = preferred_idx
    else:
        # Prefer external phone camera if available, else index 0
        phone_cams = [r[0] for r in available_cams if r[0] != 0]
        if phone_cams:
            active_idx = phone_cams[0]
        else:
            active_idx = available_cams[0][0]

    active_source_name = "Android Phone" if active_idx != 0 else "Laptop Webcam"
    print("=" * 45)
    print(f"Active camera: {active_idx}")
    print(f"Source: {active_source_name}")
    print("=" * 45)
    print("\nControls:")
    print("  [C] : Switch camera")
    print("  [Q] : Quit test")
    print("=" * 45 + "\n")

    cam_mgr = CameraManager(
        preference=str(active_idx),
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )
    cam_mgr.set_source(active_idx)

    fps = 30.0
    frame_cnt = 0
    fps_start = time.time()

    while True:
        ret, frame = cam_mgr.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        frame_cnt += 1
        if frame_cnt >= 15:
            now = time.time()
            fps = frame_cnt / (now - fps_start)
            fps_start = now
            frame_cnt = 0

        # UI Diagnostic Header
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 24, 33), -1)
        cv2.line(frame, (0, 60), (w, 60), (65, 75, 90), 2)

        source_display = "Android Phone Camera" if cam_mgr.active_source != 0 else "Laptop Webcam"
        cv2.putText(
            frame,
            f"CAMERA DIAGNOSTIC: Index #{cam_mgr.active_source} ({source_display})",
            (20, 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        phone_status = cam_mgr.get_phone_status_text()
        status_color = (80, 200, 80) if "CONNECTED" in phone_status else (40, 140, 230)
        cv2.putText(
            frame,
            f"Resolution: {w}x{h} | FPS: {fps:.1f} | Phone Status: {phone_status}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA
        )

        # Footer instructions
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 24, 33), -1)
        cv2.putText(
            frame,
            "Press [C] to Switch Camera  |  Press [Q] or [ESC] to Quit",
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Camera Diagnostic Test - Press [Q] to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            print("\n[SWITCH] Switching camera...")
            success, label = cam_mgr.switch_source()
            new_source_name = "Android Phone" if cam_mgr.active_source != 0 else "Laptop Webcam"
            print(f"Active camera: {cam_mgr.active_source}")
            print(f"Source: {new_source_name}")
            print(f"Status: {label}\n")

    cam_mgr.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")


if __name__ == "__main__":
    run_camera_test()
