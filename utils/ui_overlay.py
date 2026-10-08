"""
Exhibition-Grade UI Overlay Module for OpenCV.
Renders futuristic telemetry, confidence meters, sleek bounding boxes,
emergency alert banners, and active gesture indicators on the live camera stream.
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Dict


class ExhibitionUIOverlay:
    """
    Renders a presentation-ready Heads-Up Display (HUD) for engineering exhibitions.
    """

    def __init__(self, target_gestures: List[str], emergency_gestures: set):
        self.target_gestures = target_gestures
        self.emergency_gestures = emergency_gestures

        # Theme Colors (BGR)
        self.COLOR_BG = (18, 22, 28)
        self.COLOR_PANEL = (30, 36, 46)
        self.COLOR_BORDER = (60, 70, 85)
        self.COLOR_CYAN = (240, 210, 0)
        self.COLOR_GREEN = (90, 220, 100)
        self.COLOR_RED = (50, 50, 230)
        self.COLOR_ORANGE = (30, 140, 255)
        self.COLOR_WHITE = (245, 245, 245)
        self.COLOR_GRAY = (150, 150, 150)
        self.COLOR_DARK_TEXT = (25, 25, 25)

        self._flash_state = False
        self._last_flash_time = time.time()

    def draw_top_bar(self, frame: np.ndarray, fps: float, status_text: str = "PURE LAPTOP MODE", is_muted: bool = False, **kwargs):
        """Draws top telemetry banner with project branding and system status."""
        h, w = frame.shape[:2]
        header_h = 56

        # Allow backward-compatibility for arduino_status argument
        display_status = kwargs.get("arduino_status", status_text)

        # Semi-transparent dark header banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.COLOR_PANEL, -1)
        cv2.line(overlay, (0, header_h), (w, header_h), self.COLOR_BORDER, 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Main Title & Subtitle
        cv2.putText(
            frame,
            "AI SIGN LANGUAGE RECOGNITION",
            (20, 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            self.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "MediaPipe 21-Landmarks | 12 Gesture Classes | ISL & Speech",
            (20, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.COLOR_CYAN,
            1,
            cv2.LINE_AA
        )

        # FPS indicator
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 380, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            self.COLOR_GREEN if fps >= 20 else self.COLOR_ORANGE,
            1,
            cv2.LINE_AA
        )

        # System Status pill
        status_color = self.COLOR_GREEN if "LAPTOP" in display_status or "ACTIVE" in display_status else self.COLOR_ORANGE
        cv2.rectangle(frame, (w - 300, 14), (w - 20, 42), self.COLOR_BG, -1)
        cv2.rectangle(frame, (w - 300, 14), (w - 20, 42), status_color, 1)
        cv2.putText(
            frame,
            display_status,
            (w - 290, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            status_color,
            1,
            cv2.LINE_AA
        )

    def draw_side_gesture_panel(self, frame: np.ndarray, current_gesture: Optional[str]):
        """Draws sidebar showing list of supported 12 gestures with active highlight."""
        h, w = frame.shape[:2]
        panel_w = 160
        item_step = 28
        panel_h = len(self.target_gestures) * item_step + 40
        panel_x = w - panel_w - 15
        panel_y = 68

        # Background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.COLOR_PANEL, -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.COLOR_BORDER, 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Panel header
        cv2.putText(
            frame,
            "12 TARGET SIGNS",
            (panel_x + 12, panel_y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            self.COLOR_CYAN,
            1,
            cv2.LINE_AA
        )

        # Gesture items
        y_offset = panel_y + 44
        for g in self.target_gestures:
            is_active = (g == current_gesture)
            is_emergency = g in self.emergency_gestures

            if is_active:
                bg_color = self.COLOR_RED if is_emergency else self.COLOR_GREEN
                cv2.rectangle(frame, (panel_x + 6, y_offset - 14), (panel_x + panel_w - 6, y_offset + 8), bg_color, -1)
                text_color = self.COLOR_WHITE
                prefix = "> "
            else:
                text_color = self.COLOR_GRAY
                prefix = "  "

            cv2.putText(
                frame,
                f"{prefix}{g}",
                (panel_x + 8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                text_color,
                1,
                cv2.LINE_AA
            )
            y_offset += item_step

    def draw_hand_brackets(
        self,
        frame: np.ndarray,
        bbox: tuple,
        gesture_name: Optional[str],
        confidence: float
    ):
        """Draws aesthetic corner brackets and confidence label around the detected hand."""
        x1, y1, x2, y2 = bbox
        corner_len = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 2

        is_emergency = gesture_name in self.emergency_gestures
        color = self.COLOR_RED if is_emergency else self.COLOR_CYAN

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        if gesture_name:
            tag = f"{gesture_name} ({int(confidence * 100)}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Background pill for tag
            cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + tw + 14, y1), color, -1)
            cv2.putText(
                frame,
                tag,
                (x1 + 6, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.COLOR_WHITE,
                1,
                cv2.LINE_AA
            )

    def draw_bottom_result_banner(
        self,
        frame: np.ndarray,
        gesture_name: Optional[str],
        confidence: float,
        spoken_phrase: Optional[str]
    ):
        """Draws high-visibility bottom banner with recognized word and spoken phrase."""
        h, w = frame.shape[:2]
        banner_h = 95
        banner_y = h - banner_h

        # Semi-transparent dark bottom dock
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, banner_y), (w, h), self.COLOR_PANEL, -1)
        cv2.line(overlay, (0, banner_y), (w, banner_y), self.COLOR_BORDER, 2)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        if gesture_name:
            is_emergency = gesture_name in self.emergency_gestures
            tag_color = self.COLOR_RED if is_emergency else self.COLOR_GREEN

            # Recognition status badge
            badge_text = "EMERGENCY SIGN DETECTED" if is_emergency else "RECOGNIZED SIGN"
            cv2.putText(
                frame,
                badge_text,
                (25, banner_y + 26),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                tag_color,
                1,
                cv2.LINE_AA
            )

            # Main Large Gesture Display
            cv2.putText(
                frame,
                gesture_name,
                (25, banner_y + 68),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                self.COLOR_WHITE,
                2,
                cv2.LINE_AA
            )

            # Confidence bar
            bar_x = 240
            bar_y = banner_y + 45
            bar_w = 160
            bar_h = 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLOR_BG, -1)
            fill_w = int(bar_w * min(1.0, max(0.0, confidence)))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), tag_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLOR_BORDER, 1)
            cv2.putText(
                frame,
                f"Accuracy: {int(confidence * 100)}%",
                (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                self.COLOR_GRAY,
                1,
                cv2.LINE_AA
            )

            # Spoken Voice Feedback Subtitle
            if spoken_phrase:
                voice_label = f"Voice Audio: \"{spoken_phrase}\""
                cv2.putText(
                    frame,
                    voice_label,
                    (440, banner_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    self.COLOR_CYAN,
                    1,
                    cv2.LINE_AA
                )
        else:
            # Standby state
            cv2.putText(
                frame,
                "STATUS: AWAITING HAND GESTURE...",
                (25, banner_y + 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.60,
                self.COLOR_GRAY,
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "Position hand inside camera field of view to translate sign to speech & LCD",
                (25, banner_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                self.COLOR_WHITE,
                1,
                cv2.LINE_AA
            )

        # Footer controls
        cv2.putText(
            frame,
            "[Q] Exit   |   [R] Reset State   |   [H] Help",
            (w - 320, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.COLOR_GRAY,
            1,
            cv2.LINE_AA
        )

    def draw_emergency_flasher(self, frame: np.ndarray, gesture_name: str):
        """Flashes red border and alert banner on emergency signs (HELP / STOP)."""
        now = time.time()
        if now - self._last_flash_time > 0.35:
            self._flash_state = not self._flash_state
            self._last_flash_time = now

        if self._flash_state:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), self.COLOR_RED, 8)

            alert_text = f"*** EMERGENCY ALERT: {gesture_name} *** BUZZER & LED ACTIVE"
            text_size, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            alert_x = (w - text_size[0]) // 2
            cv2.rectangle(frame, (alert_x - 15, 65), (alert_x + text_size[0] + 15, 105), self.COLOR_RED, -1)
            cv2.putText(
                frame,
                alert_text,
                (alert_x, 93),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                self.COLOR_WHITE,
                2,
                cv2.LINE_AA
            )
