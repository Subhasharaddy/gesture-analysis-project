"""
Safe no-op mock for utils.serial_communicator.
Hardware Arduino is not connected and serial communication is disabled in this environment.
"""

import logging

logger = logging.getLogger("SerialCommunicatorStub")


class ArduinoSerialBridge:
    """Safe no-op replacement for Arduino serial bridge without pyserial dependency."""

    def __init__(self, port=None, baud_rate=115200, auto_detect=True, enable_mock=True, cooldown_seconds=1.5):
        self.port = None
        self.baud_rate = baud_rate
        self.is_connected = False
        self.is_mock_mode = True
        self.cooldown_seconds = cooldown_seconds
        self.serial_conn = None
        logger.info("ArduinoSerialBridge initialized in safe NO-OP stub mode (no hardware).")

    def get_status_text(self) -> str:
        return "NO ARDUINO (SIMULATION)"

    def send_gesture(self, gesture_name: str, is_emergency: bool = False) -> bool:
        return False

    def close(self):
        pass


SerialCommunicator = ArduinoSerialBridge

__all__ = ["ArduinoSerialBridge", "SerialCommunicator"]
