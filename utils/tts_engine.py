"""
Threaded, Non-Blocking Text-To-Speech (TTS) Engine.
Prevents webcam video freezing during audio synthesis and handles intelligent
debouncing, phrase mapping, and priority emergency voice output.
"""

import threading
import queue
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS_Engine")


class TextToSpeechWorker:
    """
    Asynchronous Text-to-Speech engine using a dedicated background thread.
    Guarantees 0-latency impact on OpenCV frame rendering.
    """

    def __init__(self, speech_map: Optional[dict] = None, cooldown_seconds: float = 3.5):
        self.speech_map = speech_map or {}
        self.cooldown_seconds = cooldown_seconds

        self._queue = queue.Queue(maxsize=10)
        self._running = True
        self._last_spoken_sign = None
        self._last_spoken_time = 0.0
        self._lock = threading.Lock()

        # Start background worker thread
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("Text-To-Speech background worker initialized.")

    def _worker_loop(self):
        """Dedicated thread executing TTS synthesis using pyttsx3."""
        engine = None
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
            
            import pyttsx3
            # Initialize SAPI5 on Windows
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)     # Natural speaking speed
            engine.setProperty("volume", 1.0)   # Max exhibition audibility

            # Try to pick a pleasant voice if available
            voices = engine.getProperty("voices")
            if voices:
                # Default to index 0 or female voice if preferred
                engine.setProperty("voice", voices[0].id)
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3 audio engine: {e}. TTS will run in dummy log mode.")
            engine = None

        while self._running:
            try:
                # Wait for next text item with timeout to allow clean shutdown
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break

                text, is_emergency = item
                logger.info(f"Speaking: '{text}' (Emergency={is_emergency})")

                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        logger.error(f"Error during pyttsx3 speech synthesis: {err}")
                        # Re-attempt engine re-init if COM got corrupted
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                        except Exception:
                            pass
                else:
                    # Simulation mode fallback
                    time.sleep(0.8)

                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Unexpected TTS loop error: {e}")

    def speak(self, gesture_name: str, is_emergency: bool = False, force: bool = False):
        """
        Submits a gesture to be spoken, applying debouncing rules.

        Args:
            gesture_name: Name of gesture (e.g. 'HELLO', 'HELP')
            is_emergency: Whether this is an urgent emergency sign
            force: If True, bypasses cooldown checks
        """
        if not gesture_name or gesture_name in ("No hand detected", "Detecting...", "STANDBY", "Unknown Gesture"):
            return

        now = time.time()
        clean_name = str(gesture_name).strip().upper()
        phrase = self.speech_map.get(clean_name, self.speech_map.get(gesture_name, gesture_name))

        try:
            with self._lock:
                # Rule 1: Allow immediately if sign changed
                sign_changed = (clean_name != self._last_spoken_sign)
                # Rule 2: Remove cooldown for non-emergency repetitive speech
                # We only speak once per new gesture, unless forced or emergency

                if force or is_emergency or sign_changed:
                    self._last_spoken_sign = clean_name
                    self._last_spoken_time = now

                    # Clear backlog so emergency OR new gestures are spoken without delay
                    # and old stale gestures are discarded
                    if is_emergency or sign_changed:
                        while not self._queue.empty():
                            try:
                                self._queue.get_nowait()
                                self._queue.task_done()
                            except (queue.Empty, ValueError):
                                break

                    logger.info(f"[TTS_DISPATCH] Speaking gesture='{clean_name}' | phrase='{phrase}' (emergency={is_emergency}, changed={sign_changed})")
                    try:
                        self._queue.put_nowait((phrase, is_emergency))
                    except queue.Full:
                        logger.warning("TTS queue full; dropping frame utterance.")
        except Exception as e:
            logger.error(f"[TTS_ERROR] Failed to queue speech for '{gesture_name}': {e}", exc_info=True)

    def stop(self):
        """Stops the worker thread cleanly."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("TTS Engine stopped.")
