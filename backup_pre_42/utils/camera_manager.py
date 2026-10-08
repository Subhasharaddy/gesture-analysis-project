"""
Smart Camera Manager for Pure Laptop AI Sign Language System.
Automatically manages video input:
  - Source A: Built-in Laptop Webcam (Index 0 via DirectShow)
  - Source B: Android Phone exposed as Windows Webcam/UVC camera (Index 1, 2, etc.)
  - Source C: Android Phone camera stream through local HTTP/MJPEG URL (e.g. http://127.0.0.1:4747/video or IP Webcam)
  - Source D: Optional ADB-based phone camera bridge
  - Seamless on-the-fly camera switching without restarting or freezing.
  - Verification of real frames (prevents ghost device stubs).
  - Resilient error handling and graceful fallback to laptop webcam.
"""

import cv2
import time
import shutil
import logging
import subprocess
import urllib.request
import socket
from typing import Tuple, Optional, Union, List, Dict

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Central Camera Abstraction for OpenCV video capture across Laptop Webcams,
    Android Phone USB cameras (UVC / DroidCam / Iriun), and local HTTP/MJPEG streams.
    Guarantees robust fallback, dynamic device discovery, and instant switching.
    """

    def __init__(
        self,
        preference: str = "laptop",
        laptop_index: int = 0,
        phone_index: int = 1,
        phone_stream_url: str = "http://127.0.0.1:4747/video",
        target_width: int = 1280,
        target_height: int = 720
    ):
        self.preference = str(preference).strip().lower()
        self.laptop_index = int(laptop_index)
        self.phone_index = int(phone_index)
        self.phone_stream_url = str(phone_stream_url).strip()
        self.target_width = target_width
        self.target_height = target_height

        self.cap: Optional[cv2.VideoCapture] = None
        self.active_source: Union[int, str] = self.laptop_index
        self.source_label: str = "INITIALIZING"
        self.available_indexes: List[int] = []
        self.device_info: List[Dict] = []
        self.consecutive_failures: int = 0
        self.adb_connected: bool = False

        # Scan for devices & connect
        self.scan_local_cameras()
        self.check_adb_bridge()
        self.connect()

    def scan_local_cameras(self, max_check: int = 4) -> List[int]:
        """
        Dynamically scans camera indexes 0 through max_check-1.
        CRITICAL: Only devices that actually yield a VALID, NON-EMPTY FRAME are registered.
        Ghost devices (where isOpened() is True but read() fails) are rejected.
        """
        found = []
        info_list = []

        logger.info(f"Scanning camera indexes 0 to {max_check - 1}...")

        for i in range(max_check):
            # If device i is already held open by self.cap, don't open a duplicate handle!
            if self.cap and self.cap.isOpened() and i == self.active_source:
                found.append(i)
                is_phone = (i != self.laptop_index)
                label = f"Laptop Webcam (Index {i})" if not is_phone else f"Android Phone Camera (USB Index {i})"
                info_list.append({
                    "index": i,
                    "label": label,
                    "is_phone": is_phone,
                    "resolution": (self.target_width, self.target_height),
                    "functional": True,
                    "backend": cv2.CAP_DSHOW
                })
                continue

            # Try DirectShow first on Windows (fastest, most compatible)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            used_backend = cv2.CAP_DSHOW
            opened = cap.isOpened()
            has_frame = False
            w, h = 0, 0

            if opened:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    has_frame = True
                cap.release()
                time.sleep(0.15)

            # If DirectShow did not yield a valid frame, try default/MSMF only for index 0
            if not has_frame and i == 0:
                cap_alt = cv2.VideoCapture(i)
                if cap_alt.isOpened():
                    ret_alt, frame_alt = cap_alt.read()
                    if ret_alt and frame_alt is not None and frame_alt.size > 0:
                        h, w = frame_alt.shape[:2]
                        has_frame = True
                        used_backend = cv2.CAP_ANY
                cap_alt.release()
                time.sleep(0.15)

            if has_frame:
                found.append(i)
                is_phone = (i != self.laptop_index)
                label = f"Laptop Webcam (Index {i})" if not is_phone else f"Android Phone Camera (USB Index {i})"
                info_list.append({
                    "index": i,
                    "label": label,
                    "is_phone": is_phone,
                    "resolution": (w, h),
                    "functional": True,
                    "backend": used_backend
                })
                logger.info(f"Detected working camera at index {i}: {w}x{h} ({label})")

        self.available_indexes = found
        self.device_info = info_list

        # If an external camera was discovered, update default phone_index dynamically
        external_cams = [d["index"] for d in info_list if d["is_phone"]]
        if external_cams:
            self.phone_index = external_cams[0]
            logger.info(f"Dynamically registered external Android Phone camera at index {self.phone_index}")

        logger.info(f"Discovered {len(found)} active camera(s): {found}")
        return found

    def check_adb_bridge(self) -> bool:
        """
        Checks if ADB is available and an Android phone is attached via USB cable.
        If port 4747 is needed (e.g. DroidCam / IP Webcam), automatically sets up port forwarding.
        """
        adb_path = shutil.which("adb")
        if not adb_path:
            self.adb_connected = False
            return False

        try:
            res = subprocess.run(
                ["adb", "devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.startswith("List of")]
            devices = [l.split()[0] for l in lines if "\tdevice" in l]

            if devices:
                self.adb_connected = True
                logger.info(f"ADB detected {len(devices)} connected Android device(s): {devices}")
                try:
                    subprocess.run(
                        ["adb", "forward", "tcp:4747", "tcp:4747"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=2
                    )
                except Exception:
                    pass
                return True
        except Exception as e:
            logger.debug(f"ADB check ignored: {e}")

        self.adb_connected = False
        return False

    def is_phone_connected(self) -> bool:
        """
        Returns True if an Android phone camera is detected via USB webcam index,
        ADB bridge, or active stream URL.
        """
        # 1. External USB camera index detected with valid frames
        for d in self.device_info:
            if d.get("is_phone") and d.get("index") in self.available_indexes:
                return True

        # 2. Currently connected to phone
        if "PHONE" in self.source_label.upper() and self.is_opened():
            return True

        # 3. Check if HTTP stream is responding
        if self._test_stream_url(self.phone_stream_url):
            return True

        # 4. ADB connected
        if self.adb_connected:
            return True

        return False

    def get_phone_status_text(self) -> str:
        """Returns clear status string: 'PHONE CONNECTED' or 'PHONE NOT DETECTED'."""
        return "PHONE CONNECTED" if self.is_phone_connected() else "PHONE NOT DETECTED"

    def _test_stream_url(self, url: str) -> bool:
        """Rapidly tests if the stream host and port are listening without hanging."""
        if not url or not url.startswith("http"):
            return False
        try:
            from urllib.parse import urlparse
            p = urlparse(url)
            host = p.hostname or "127.0.0.1"
            port = p.port or (443 if p.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except Exception:
            return False

    def _open_device(self, source: Union[int, str], backend=None) -> Optional[cv2.VideoCapture]:
        """
        Attempts to open a capture device with DirectShow backend on Windows.
        Verifies real frame capture and configures optimal buffersize with retry logic.
        """
        try:
            if isinstance(source, int):
                # On Windows, DirectShow is fastest and avoids MSMF deadlocks
                b = backend if backend is not None else cv2.CAP_DSHOW
                cap = None
                for attempt in range(3):
                    try:
                        cap = cv2.VideoCapture(source, b)
                        if (not cap or not cap.isOpened()) and b == cv2.CAP_DSHOW:
                            time.sleep(0.2)
                            cap = cv2.VideoCapture(source)
                        if cap and cap.isOpened():
                            break
                    except Exception as ex:
                        logger.debug(f"Attempt {attempt + 1} opening source {source} notice: {ex}")
                        time.sleep(0.25)
                        cap = None

                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Attempt to set requested resolution if supported
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 30)

                        ret2, frame2 = cap.read()
                        if ret2 and frame2 is not None and frame2.size > 0:
                            self.consecutive_failures = 0
                            return cap
                        else:
                            # If resolution override failed, recreate with native resolution
                            cap.release()
                            time.sleep(0.15)
                            cap = cv2.VideoCapture(source, b)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            ret3, frame3 = cap.read()
                            if ret3 and frame3 is not None and frame3.size > 0:
                                self.consecutive_failures = 0
                                return cap
                    cap.release()
                    time.sleep(0.1)
            else:
                # String stream URL (e.g. DroidCam / IP Webcam)
                url = str(source).strip()
                if not self._test_stream_url(url):
                    logger.warning(f"Stream URL {url} is not responding.")
                    return None

                cap = cv2.VideoCapture(url)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.consecutive_failures = 0
                        return cap
                    cap.release()
        except Exception as e:
            logger.warning(f"Error opening source {source}: {e}")

        return None

    def connect(self, source: Optional[Union[int, str]] = None) -> bool:
        """
        Connects to the desired camera source, or uses preference with automatic graceful fallback.
        Never crashes the application if camera is unavailable.
        """
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if source is not None:
            return self.set_source(source)[0]

        pref = self.preference.lower()

        # Prioritize Laptop Webcam if preference is "laptop" or default
        if pref in ("laptop", "auto"):
            logger.info(f"Opening Laptop Webcam (Index {self.laptop_index}) via DirectShow...")
            cap = self._open_device(self.laptop_index, backend=cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                self.cap = cap
                self.active_source = self.laptop_index
                self.source_label = f"CAMERA: LAPTOP WEBCAM | STATUS: CONNECTED | INDEX: {self.laptop_index}"
                logger.info(f"Connected: {self.source_label}")
                return True

            # If index 0 failed, check any other available index
            for idx in self.available_indexes:
                if idx != self.laptop_index:
                    cap_alt = self._open_device(idx, backend=cv2.CAP_DSHOW)
                    if cap_alt and cap_alt.isOpened():
                        self.cap = cap_alt
                        self.active_source = idx
                        self.source_label = f"CAMERA: WEBCAM (Index {idx})"
                        return True

            self.source_label = "CAMERA: LAPTOP WEBCAM | STATUS: NOT DETECTED"
            logger.error("Failed to connect to Laptop Webcam.")
            return False

        elif pref == "phone":
            logger.info("Connecting to Android Phone Camera...")
            external_cams = [d["index"] for d in self.device_info if d["is_phone"]]
            target_phone_idx = external_cams[0] if external_cams else self.phone_index

            if target_phone_idx in self.available_indexes:
                cap = self._open_device(target_phone_idx)
                if cap and cap.isOpened():
                    self.cap = cap
                    self.active_source = target_phone_idx
                    self.source_label = f"CAMERA: PHONE CAMERA | STATUS: CONNECTED | USB INDEX: {target_phone_idx}"
                    logger.info(f"Connected: {self.source_label}")
                    return True

            # Test stream URL
            if self._test_stream_url(self.phone_stream_url):
                cap = self._open_device(self.phone_stream_url)
                if cap and cap.isOpened():
                    self.cap = cap
                    self.active_source = self.phone_stream_url
                    self.source_label = "CAMERA: PHONE CAMERA | STATUS: CONNECTED | STREAM URL"
                    return True

            logger.warning("Phone Camera unavailable. Falling back to Laptop Webcam...")
            cap = self._open_device(self.laptop_index, backend=cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                self.cap = cap
                self.active_source = self.laptop_index
                self.source_label = f"CAMERA: LAPTOP WEBCAM | STATUS: CONNECTED | INDEX: {self.laptop_index}"
                return True

        self.source_label = "CAMERA: LAPTOP WEBCAM | STATUS: NOT DETECTED"
        return False

    def reconnect(self) -> bool:
        """Attempts to reconnect to the camera hardware."""
        logger.info("Attempting camera reconnect...")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.scan_local_cameras()
        return self.connect(self.active_source)

    def switch_source(self) -> Tuple[bool, str]:
        """
        Toggles live between Laptop Webcam and Android Phone Camera.
        If the target phone camera is not streaming, preserves the laptop webcam.
        Returns (success_bool, new_label_str).
        """
        old_source = self.active_source

        # If currently on laptop camera, switch to phone
        if (isinstance(old_source, int) and old_source == self.laptop_index) or "LAPTOP" in self.source_label.upper():
            # Fast rescan in case phone was just plugged in
            self.scan_local_cameras()
            external_cams = [d["index"] for d in self.device_info if d["is_phone"]]
            target = external_cams[0] if external_cams else None

            # 1. Try external USB camera index if functional
            if target is not None:
                new_cap = self._open_device(target)
                if new_cap and new_cap.isOpened():
                    if self.cap and self.cap.isOpened():
                        self.cap.release()
                    self.cap = new_cap
                    self.active_source = target
                    self.source_label = f"CAMERA: PHONE CAMERA | STATUS: CONNECTED | USB INDEX: {target}"
                    return True, self.source_label

            # 2. Try phone stream URL if active
            if self._test_stream_url(self.phone_stream_url):
                new_cap = self._open_device(self.phone_stream_url)
                if new_cap and new_cap.isOpened():
                    if self.cap and self.cap.isOpened():
                        self.cap.release()
                    self.cap = new_cap
                    self.active_source = self.phone_stream_url
                    self.source_label = "CAMERA: PHONE CAMERA | STATUS: CONNECTED | STREAM URL"
                    return True, self.source_label

            # Phone is not available: KEEP laptop camera running!
            logger.warning("Phone camera not detected or not streaming. Keeping Laptop Webcam active.")
            return False, "Phone camera not detected / not streaming"

        else:
            # Currently on phone, switch back to laptop webcam
            if self.cap and self.cap.isOpened():
                self.cap.release()

            new_cap = self._open_device(self.laptop_index, backend=cv2.CAP_DSHOW)
            if new_cap and new_cap.isOpened():
                self.cap = new_cap
                self.active_source = self.laptop_index
                self.source_label = f"CAMERA: LAPTOP WEBCAM | STATUS: CONNECTED | INDEX: {self.laptop_index}"
                return True, self.source_label
            else:
                self.source_label = "CAMERA: LAPTOP WEBCAM | STATUS: NOT DETECTED"
                return False, self.source_label

    def set_source(self, source: Union[int, str]) -> Tuple[bool, str]:
        """
        Explicitly selects and activates a camera device by index (0, 1, 2...),
        URL ('http://...'), or keyword ('phone', 'laptop', 'auto').
        """
        if isinstance(source, str):
            s_lower = source.strip().lower()
            if s_lower == "laptop":
                source = self.laptop_index
            elif s_lower == "phone":
                self.scan_local_cameras()
                external_cams = [d["index"] for d in self.device_info if d["is_phone"]]
                if external_cams:
                    source = external_cams[0]
                elif self._test_stream_url(self.phone_stream_url):
                    source = self.phone_stream_url
                else:
                    return False, "Phone camera not detected / not streaming"
            elif s_lower == "auto":
                return self.connect(), self.source_label

        # If switching to same source and already open, return True
        if self.cap and self.cap.isOpened() and self.active_source == source:
            return True, self.source_label

        new_cap = self._open_device(source)
        if new_cap and new_cap.isOpened():
            if self.cap and self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = new_cap
            self.active_source = source

            if source == self.laptop_index:
                self.source_label = f"CAMERA: LAPTOP WEBCAM | STATUS: CONNECTED | INDEX: {self.laptop_index}"
            elif isinstance(source, int):
                self.source_label = f"CAMERA: PHONE CAMERA | STATUS: CONNECTED | USB INDEX: {source}"
            else:
                self.source_label = f"CAMERA: PHONE CAMERA | STATUS: CONNECTED | STREAM URL"
            return True, self.source_label

        return False, f"Could not connect to {source}"

    def set_phone_stream_url(self, url: str) -> Tuple[bool, str]:
        """
        Connects to a custom phone stream URL (e.g., IP Webcam or DroidCam).
        """
        url = str(url).strip()
        if not url.startswith("http"):
            return False, "URL must start with http:// or https://"

        self.phone_stream_url = url
        return self.set_source(url)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Reads a frame from the active camera.
        If frame capture repeatedly fails (e.g. phone unplugged), automatically
        falls back to laptop webcam without crashing.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.consecutive_failures = 0
                return True, frame
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 15:
                    logger.warning("Repeated frame read failures. Attempting fallback to Laptop Webcam...")
                    self.consecutive_failures = 0
                    if self.active_source != self.laptop_index:
                        self.set_source(self.laptop_index)
        return False, None

    def is_opened(self) -> bool:
        """Returns True if capture device is opened and ready."""
        return bool(self.cap and self.cap.isOpened())

    def release(self):
        """Releases the camera hardware cleanly."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Camera device released.")
