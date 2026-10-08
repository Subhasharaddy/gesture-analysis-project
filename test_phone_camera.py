"""
Dedicated Phone Camera Test Script for AI Sign Language Recognition.
Connect Android Phone via USB data cable, select Webcam or start DroidCam/Iriun,
and test the phone camera feed with real-time HUD and camera switching.
"""

import cv2
import time
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import config
from utils.camera_manager import CameraManager


def main():
    print("=" * 60)
    print("           ANDROID PHONE USB CAMERA TEST")
    print("=" * 60)
    print("[INFO] Scanning cameras...")

    cam_mgr = CameraManager(
        preference=config.CAMERA_PREFERENCE,
        laptop_index=config.LAPTOP_CAMERA_INDEX,
        phone_index=config.PHONE_CAMERA_INDEX,
        phone_stream_url=config.PHONE_STREAM_URL,
        target_width=config.CAMERA_WIDTH,
        target_height=config.CAMERA_HEIGHT
    )

    # Print scan report in requested format
    for dev in cam_mgr.device_info:
        idx = dev["index"]
        if idx == config.LAPTOP_CAMERA_INDEX:
            print(f"[OK] Laptop Webcam detected at index {idx}")
        else:
            print(f"[OK] Android Phone detected at index {idx}")

    phone_detected = cam_mgr.is_phone_connected()
    if phone_detected:
        print("[STATUS] PHONE CONNECTED (USB)")
    else:
        print("[STATUS] PHONE NOT DETECTED (Using Laptop Webcam as fallback)")

    print("\nStarting video stream...")
    print("PHONE CAMERA TEST")
    print("Press Q to exit")
    print("Press C to switch camera")
    print("=" * 60 + "\n")

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

        # Draw HUD
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 24, 33), -1)
        cv2.line(frame, (0, 60), (w, 60), (65, 75, 90), 2)

        is_phone = "PHONE" in cam_mgr.source_label.upper()
        header_title = "PHONE CAMERA TEST (ACTIVE)" if is_phone else "LAPTOP WEBCAM TEST (PHONE FALLBACK)"
        header_color = (0, 215, 255) if is_phone else (255, 255, 255)

        cv2.putText(frame, header_title, (20, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, header_color, 1, cv2.LINE_AA)

        phone_text = cam_mgr.get_phone_status_text()
        phone_color = (80, 200, 80) if "CONNECTED" in phone_text else (40, 140, 230)
        cv2.putText(
            frame,
            f"Source: {cam_mgr.source_label} | {phone_text} | FPS: {fps:.1f}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            phone_color,
            1,
            cv2.LINE_AA
        )

        # Bottom banner
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 24, 33), -1)
        cv2.putText(
            frame,
            "Press [C] to Switch Camera  |  Press [Q] to Exit",
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Phone Camera Test - Press [Q] to Exit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            print("[SWITCH] Switching camera source...")
            success, label = cam_mgr.switch_source()
            print(f"[SWITCH] Current active camera: {label}")

    cam_mgr.release()
    cv2.destroyAllWindows()
    print("Phone camera test finished.")


if __name__ == "__main__":
    main()
