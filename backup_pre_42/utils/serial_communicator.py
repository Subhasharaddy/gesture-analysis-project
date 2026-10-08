"""
Arduino Serial Communication Module.
Handles automatic COM port discovery, robust bidirectional serial communication,
debounced command sending, auto-reconnection, and fallback Mock/Simulation mode.
"""

import time
import threading
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SerialCommunicator")


class ArduinoSerialBridge:
    """
    Manages serial communication with Arduino UNO running exhibition firmware.
    Provides non-blocking sends, auto-detection, and mock simulation mode.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 9600,
        auto_detect: bool = True,
        enable_mock: bool = True,
        cooldown_seconds: float = 1.2
    ):
        self.requested_port = port
        self.baud_rate = baud_rate
        self.auto_detect = auto_detect
        self.enable_mock = enable_mock
        self.cooldown_seconds = cooldown_seconds

        self.serial_conn = None
        self.active_port = None
        self.is_connected = False
        self.is_mock_mode = False

        self._last_sent_sign = None
        self._last_sent_time = 0.0
        self._lock = threading.Lock()

        # Connect on init
        self.connect()

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns a list of available serial port names."""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            return [p.device for p in ports]
        except Exception:
            return []

    def detect_arduino_port(self) -> Optional[str]:
        """
        Scans COM ports to find official Arduino or CH340 USB-Serial boards.
        """
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                # Check known Arduino signatures and clone USB chips (CH340, CP2102, FTDI)
                if any(k in desc for k in ["arduino", "ch340", "usb serial", "usb-to-serial"]):
                    logger.info(f"Detected likely Arduino device on {p.device}: {p.description}")
                    return p.device
                # Check common Arduino USB VID:PID
                if "2341:" in hwid or "1a86:" in hwid or "10c4:" in hwid:
                    logger.info(f"Detected Arduino hardware ID on {p.device}: {p.hwid}")
                    return p.device

            # If no explicit match but ports exist, return the first available port
            if ports:
                logger.info(f"No explicit Arduino signature, selecting first available port: {ports[0].device}")
                return ports[0].device

        except Exception as e:
            logger.warning(f"Error enumerating serial ports: {e}")
        return None

    def connect(self) -> bool:
        """Establishes connection to Arduino or falls back to Mock mode."""
        port_to_try = self.requested_port
        if not port_to_try and self.auto_detect:
            port_to_try = self.detect_arduino_port()

        if port_to_try:
            try:
                import serial
                logger.info(f"Opening serial connection on {port_to_try} at {self.baud_rate} baud...")
                self.serial_conn = serial.Serial(
                    port=port_to_try,
                    baudrate=self.baud_rate,
                    timeout=1.0,
                    write_timeout=1.0
                )
                self.active_port = port_to_try
                self.is_connected = True
                self.is_mock_mode = False

                # Allow Arduino UNO 1.5 - 2.0 seconds to finish DTR auto-reset
                time.sleep(1.8)

                # Flush any startup bootloader chatter
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                logger.info(f"Successfully connected to Arduino on {port_to_try}.")
                return True

            except Exception as e:
                logger.warning(f"Failed to connect to hardware on {port_to_try}: {e}")

        # Fallback to Mock/Simulation mode
        if self.enable_mock:
            self.is_connected = True
            self.is_mock_mode = True
            self.active_port = "VIRTUAL_SIMULATION"
            logger.info("Operating in VIRTUAL ARDUINO SIMULATION MODE. Hardware signals logged to console.")
            return True

        self.is_connected = False
        return False

    def send_gesture(self, gesture_name: str, is_emergency: bool = False, force: bool = False) -> bool:
        """
        Sends formatted gesture string to Arduino:
        Protocol: "G:<GESTURE_NAME>\n"
        Example: "G:HELLO\n", "G:HELP\n"
        """
        now = time.time()
        with self._lock:
            # Send immediately if sign changed, or if cooldown elapsed, or if forced/emergency
            sign_changed = (gesture_name != self._last_sent_sign)
            cooldown_passed = (now - self._last_sent_time >= self.cooldown_seconds)

            if not (force or is_emergency or sign_changed or cooldown_passed):
                return False

            self._last_sent_sign = gesture_name
            self._last_sent_time = now

            command = f"G:{gesture_name.strip().upper()}\n"

            if self.is_mock_mode:
                alert_tag = " [EMERGENCY ALERT TRIGGERED]" if is_emergency else ""
                logger.info(f"[SIMULATED ARDUINO LCD/BUZZER] Received: '{command.strip()}'{alert_tag}")
                return True

            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.write(command.encode("ascii"))
                    self.serial_conn.flush()
                    return True
                except Exception as e:
                    logger.error(f"Serial write error: {e}. Attempting reconnect...")
                    self.is_connected = False
                    self.connect()
                    return False

        return False

    def get_status_text(self) -> str:
        """Returns short status string for UI display."""
        if not self.is_connected:
            return "ARDUINO: DISCONNECTED"
        if self.is_mock_mode:
            return "ARDUINO: SIMULATION MODE"
        return f"ARDUINO: {self.active_port} (ACTIVE)"

    def close(self):
        """Closes the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("Arduino Serial Bridge closed.")
