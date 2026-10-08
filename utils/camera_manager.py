"""
Rock-Solid Camera Manager for AI Sign Language Recognition.
Safely detects, initializes, and manages camera hardware without freezing,
driver deadlocks, ghost devices, or multiple simultaneous camera handles.
"""

import cv2
import time
import socket
import logging
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np

import config

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Safe, high-performance Camera Abstraction for OpenCV VideoCapture.
    Guarantees:
      - Sequential 0 -> 1 -> 2 -> 3 auto-discovery with zero duplicate handles
      - Clean single-camera ownership (never opens multiple cameras simultaneously)
      - Fast 640x480 @ 30 FPS with CAP_PROP_BUFFERSIZE = 1 (zero frame lag)
      - Non-crashing read() with fast clamping
      - Debounced recovery mechanism for unplugged/reconnected devices
    """

    def __init__(
        self,
        preference: str = "laptop",
        laptop_index: int = 0,
        phone_index: int = 1,
        phone_stream_url: str = "http://127.0.0.1:4747/video",
        target_width: int = 640,
        target_height: int = 480
    ):
        self.preference = str(preference).strip().lower()
        self.laptop_index = int(laptop_index)
        self.phone_index = int(phone_index)
        self.phone_stream_url = str(phone_stream_url).strip()
        self.target_width = getattr(config, "CAMERA_WIDTH", target_width)
        self.target_height = getattr(config, "CAMERA_HEIGHT", target_height)

        self.cap: Optional[cv2.VideoCapture] = None
        self.active_source: Union[int, str] = self.laptop_index
        self.source_label: str = "INITIALIZING"
        self.available_indexes: List[int] = []
        self.consecutive_failures: int = 0
        self.last_recovery_time: float = 0.0

        # Safe sequential initialization
        self.init_camera()

    def init_camera(self) -> bool:
        """
        Safely tests camera indexes sequentially:
          Try camera 0 -> If unavailable -> try camera 1 -> If unavailable -> try camera 2 -> try camera 3
        Retains the first working camera without reopening or closing it.
        """
        self.release()

        # Build candidate search order
        candidate_indices = [self.laptop_index]
        for idx in [0, 1, 2, 3]:
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        logger.info(f"Scanning camera indexes sequentially: {candidate_indices}")

        for idx in candidate_indices:
            # Try DSHOW first on Windows (standard for integrated webcams), then fallback to default
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue

                if not cap.isOpened():
                    cap.release()
                    continue

                # 2. Configure 640x480, 30 FPS, buffer=1
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                # 3. Test frame read to verify actual frame delivery
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.target_height
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

                    # Working camera found! Keep it open and store handle.
                    self.cap = cap
                    self.active_source = idx
                    if idx not in self.available_indexes:
                        self.available_indexes.append(idx)

                    cam_name = "Laptop Webcam" if idx == 0 else f"Camera Index {idx}"
                    self.source_label = f"Camera index: {idx} | CONNECTED | {actual_w}x{actual_h} | {actual_fps} FPS"

                    print(f"Camera index: {idx}")
                    print(f"Camera status: CONNECTED")
                    print(f"Resolution: {actual_w}x{actual_h}")
                    print(f"FPS: {actual_fps}")
                    return True
                else:
                    cap.release()

        # None of the camera indices worked
        print("No working camera detected.")
        self.cap = None
        self.source_label = "No working camera detected."
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the latest frame from the camera safely.
        Never crashes if camera is closed or frame read fails.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            self.consecutive_failures += 1
            return False, None

        self.consecutive_failures = 0
        # Fast clamping if camera native mode exceeds target size
        if frame.shape[1] > self.target_width or frame.shape[0] > self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        return True, frame

    def is_opened(self) -> bool:
        """Returns True if camera is open and active."""
        return bool(self.cap is not None and self.cap.isOpened())

    def release(self):
        """Cleanly releases the camera handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Camera released.")

    def recover(self) -> bool:
        """
        Attempts to recover a lost camera gracefully without rapid re-open loops.
        """
        now = time.time()
        if now - self.last_recovery_time < 2.0:
            return False
        self.last_recovery_time = now

        logger.info("Attempting graceful camera recovery...")
        self.release()
        time.sleep(0.25)
        return self.init_camera()

    def reconnect(self) -> bool:
        """Manual reconnect triggered by user."""
        return self.recover()

    def switch_source(self) -> Tuple[bool, str]:
        """
        Switches between detected camera sources (e.g. index 0 <-> index 1).
        Releases old camera BEFORE opening new camera.
        """
        if not isinstance(self.active_source, int):
            return self.recover(), self.source_label

        target_idx = 1 if self.active_source == 0 else 0
        logger.info(f"Switching from camera {self.active_source} to {target_idx}...")

        # 1. Release active camera FIRST
        self.release()
        time.sleep(0.2)

        # 2. Try opening target index
        cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                self.cap = cap
                self.active_source = target_idx
                self.source_label = f"Camera index: {target_idx} | CONNECTED"
                return True, f"Switched to Camera {target_idx}"
            cap.release()

        # 3. Fallback to original
        recovered = self.init_camera()
        return recovered, self.source_label

    def get_diagnostic_dict(self) -> Dict[str, Union[str, int]]:
        """Returns structured camera diagnostic status."""
        return {
            "index": self.active_source if self.is_opened() else -1,
            "status": "OK" if self.is_opened() else "ERROR",
            "resolution": f"{self.target_width}x{self.target_height}",
            "fps": 30,
            "label": self.source_label
        }

    @property
    def device_info(self) -> List[Dict[str, Any]]:
        """List of detected camera devices."""
        devs = []
        for idx in self.available_indexes:
            label = "Laptop Webcam" if idx == self.laptop_index else f"Android Phone / Camera {idx}"
            devs.append({"index": idx, "label": label, "status": "CONNECTED"})
        if not devs and self.is_opened():
            idx = self.active_source if isinstance(self.active_source, int) else 0
            devs.append({"index": idx, "label": "Laptop Webcam", "status": "CONNECTED"})
        return devs

    def is_phone_connected(self) -> bool:
        """Checks if a secondary phone camera (e.g. index 1 or stream) is connected."""
        return any(idx != self.laptop_index for idx in self.available_indexes)

    def get_phone_status_text(self) -> str:
        """Returns human-readable phone camera status string."""
        if self.is_phone_connected():
            return "PHONE CONNECTED (USB)"
        return "PHONE NOT DETECTED"
