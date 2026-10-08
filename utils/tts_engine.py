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
        now = time.time()
        phrase = self.speech_map.get(gesture_name, gesture_name)

        with self._lock:
            # Rule 1: Allow immediately if sign changed
            sign_changed = (gesture_name != self._last_spoken_sign)
            # Rule 2: Allow if cooldown elapsed
            cooldown_passed = (now - self._last_spoken_time >= self.cooldown_seconds)

            if force or is_emergency or sign_changed or cooldown_passed:
                self._last_spoken_sign = gesture_name
                self._last_spoken_time = now

                # If emergency, clear backlog so emergency is spoken without delay
                if is_emergency:
                    while not self._queue.empty():
                        try:
                            self._queue.get_nowait()
                            self._queue.task_done()
                        except (queue.Empty, ValueError):
                            break

                try:
                    self._queue.put_nowait((phrase, is_emergency))
                except queue.Full:
                    logger.warning("TTS queue full; dropping frame utterance.")

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
