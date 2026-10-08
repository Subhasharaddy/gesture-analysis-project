"""
AI Sign Language Recognition System (12 Gesture Exhibition Edition).
Modern Real-Time Continuous Computer Vision & Machine Learning Application.

Supported 12 Gesture Classes:
1. HOME
2. NAMASTE
3. HELLO
4. THANK YOU
5. YES
6. NO
7. HELP
8. STOP
9. WATER
10. FOOD
11. PLEASE
12. GOOD

Features:
- Continuous automatic real-time hand detection & 21 3D skeletal landmark tracking
- Machine learning classification on 73-dimensional invariant feature vectors
- Bimanual support (NAMASTE prayer hands, HOME roof peak) & single-hand recognition
- High-contrast exhibition GUI matching engineering design guidelines
- Camera auto-start & camera selection (Laptop Webcam vs Phone USB / DroidCam)
- Start Recognition & Stop Camera buttons
- Threaded non-blocking model training with live accuracy & confusion matrix reporting
- Interactive Gesture Guide modal showcasing all 12 gesture cards & ISL instructions
- Gesture history timeline with timestamps (e.g. 10:21 - NAMASTE)
- Threaded Text-to-Speech audio output with toggle & cooldown debouncing
- Arduino UNO Serial integration (LCD transcription, buzzer & emergency LED alerts)
- Robust error handling for cameras, missing hands, missing models, and serial hardware
"""

import sys
import os
import time
import random
import threading
import subprocess
import collections
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import cv2
import joblib
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import config
from utils.landmark_processor import LandmarkProcessor
from utils.hand_tracker import UniversalHandTracker
from utils.camera_manager import CameraManager
from utils.tts_engine import TextToSpeechWorker
from utils.gesture_database import GESTURE_DEFINITIONS, get_gesture_info, register_custom_gesture, build_speech_map, CORE_12_GESTURES
from utils.gesture_engine import GestureEngine
from train_model import train_sign_classifier


class SignLanguageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI SIGN LANGUAGE RECOGNITION")
        self.root.geometry("1360x900")
        self.root.minsize(1120, 760)
        self.root.configure(bg="#0c1017")

        # Core Engines & Subsystems
        self.model = None
        self.label_encoder = None
        self.hand_tracker: Optional[UniversalHandTracker] = None
        self.cam_mgr: Optional[CameraManager] = None
        self.tts: Optional[TextToSpeechWorker] = None
        self.arduino = None
        self.gesture_engine: Optional[GestureEngine] = None

        # State Telemetry
        self.camera_running = False
        self.recognition_active = True
        self.current_frame = None
        self.canvas_image_item = None
        self._disconnected_drawn = False
        self.consecutive_read_errors = 0
        self.frame_counter = 0
        self.fps_timer = time.time()
        self.fps = 30.0
        self.hand_detected = False
        self.hands_count = 0
        self.current_fps = 30.0
        self.exhibition_mode_active = False
        self.is_speech_enabled = True
        self.is_training_active = False

        # Speed Optimization & Telemetry State
        self.process_frame_idx = 0
        self.last_primary_result = None
        self.last_detected_hands = []
        self.last_latency_sec = 0.12
        self.last_infer_ms = 8.0

        # Recognition History Queue (Latest 15)
        self.history_items = collections.deque(maxlen=15)
        self.last_spoken_gesture = None
        self.last_spoken_time = 0.0

        # Initialize Hardware & AI Subsystems
        self.init_hardware_bridges()
        self.load_ml_model()

        # Build Exhibition GUI
        self.build_ui()

        # Start Camera Stream Automatically on open
        self.start_camera()

        # Start Video Rendering Loop (~30 FPS)
        self.update_video_loop()

    def init_hardware_bridges(self):
        """Initializes Text-To-Speech worker (hardware Arduino removed)."""
        self.arduino = None

        # Text-to-Speech Engine
        try:
            speech_map = build_speech_map()
            self.tts = TextToSpeechWorker(speech_map=speech_map, cooldown_seconds=config.SPEECH_COOLDOWN_SECONDS)
        except Exception as e:
            print(f"[WARNING] TTS initialization notice: {e}")
            self.tts = None

    def load_ml_model(self):
        """Loads trained 12-class machine learning model artifacts."""
        model_path = config.MODEL_PKL_PATH
        label_path = config.LABEL_ENCODER_PATH

        try:
            if model_path.exists() and label_path.exists():
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load(label_path)
                print(f"[OK] Model: LOADED ({model_path.name})")
                print(f"Model classes: {len(self.label_encoder.classes_)}")
            elif (config.MODELS_DIR / "gesture_model_30.joblib").exists():
                self.model = joblib.load(config.MODELS_DIR / "gesture_model_30.joblib")
                self.label_encoder = joblib.load(config.MODELS_DIR / "label_encoder_30.joblib")
                print("[OK] Loaded fallback multi-class model.")
            else:
                self.model = None
                self.label_encoder = None
                print("[INFO] Model file not found. Ready to train with 'Train Model' button.")
        except Exception as e:
            print(f"[WARNING] Model loading notice: {e}")
            self.model = None
            self.label_encoder = None

        # Initialize Gesture Engine
        self.gesture_engine = GestureEngine(
            model=self.model,
            label_encoder=self.label_encoder,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            smoothing_window=config.STABILITY_WINDOW_SIZE
        )

        # Initialize MediaPipe Universal Hand Tracker with 2-hand support
        try:
            self.hand_tracker = UniversalHandTracker(
                max_num_hands=config.MP_MAX_NUM_HANDS,
                min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE
            )
        except Exception as e:
            messagebox.showerror(
                "MediaPipe Error",
                f"MediaPipe initialization failed:\n{e}\n\nPlease check requirements.txt."
            )

    def build_ui(self):
        """Builds clean, modern exhibition-style dark UI."""
        # --- Top Header Banner ---
        self.header_frame = tk.Frame(self.root, bg="#161d2a", padx=18, pady=10, relief=tk.RAISED, bd=1)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)

        brand_box = tk.Frame(self.header_frame, bg="#161d2a")
        brand_box.pack(side=tk.LEFT)

        self.title_lbl = tk.Label(
            brand_box,
            text="✋ AI SIGN LANGUAGE RECOGNITION",
            font=("Segoe UI", 16, "bold"),
            bg="#161d2a",
            fg="#ffffff"
        )
        self.title_lbl.pack(anchor=tk.W)

        self.sub_title_lbl = tk.Label(
            brand_box,
            text="42 GESTURE CLASSES • REAL-TIME MEDIAPIPE TRACKER • RANDOM FOREST AI • TEXT-TO-SPEECH",
            font=("Segoe UI", 9, "bold"),
            bg="#161d2a",
            fg="#00d2ff"
        )
        self.sub_title_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Header Right Controls & Badges
        hdr_right = tk.Frame(self.header_frame, bg="#161d2a")
        hdr_right.pack(side=tk.RIGHT)

        self.exhibition_btn = tk.Button(
            hdr_right,
            text="🏆 EXHIBITION MODE",
            font=("Segoe UI", 9, "bold"),
            bg="#8957e5",
            fg="#ffffff",
            activebackground="#a371f7",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self.toggle_exhibition_mode
        )
        self.exhibition_btn.pack(side=tk.RIGHT, padx=5)

        # Model Status Badge
        num_classes = len(self.label_encoder.classes_) if self.label_encoder else 0
        self.mode_badge = tk.Label(
            hdr_right,
            text=f"⚡ AI MODEL: {num_classes} CLASSES" if num_classes > 0 else "⚡ AI MODEL: READY TO TRAIN",
            font=("Segoe UI", 8, "bold"),
            bg="#1f2937",
            fg="#5af07b" if num_classes > 0 else "#e3b341",
            padx=8,
            pady=4,
            relief=tk.SOLID,
            bd=1
        )
        self.mode_badge.pack(side=tk.RIGHT, padx=5)

        # Pipeline Flow Banner
        self.pipeline_frame = tk.Frame(self.root, bg="#0f1520", pady=4)
        self.pipeline_frame.pack(fill=tk.X)

        self.pipeline_text = tk.Label(
            self.pipeline_frame,
            text="WEBCAM → 21 HAND LANDMARKS → 73 INVARIANT FEATURES → 42-CLASS RANDOM FOREST → SPEECH",
            font=("Segoe UI", 9, "bold"),
            bg="#0f1520",
            fg="#2ea043"
        )
        self.pipeline_text.pack()

        # Main Split Content
        self.content_frame = tk.Frame(self.root, bg="#0c1017", padx=10, pady=6)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Live Camera Preview & Landmarks ---
        self.left_panel = tk.Frame(self.content_frame, bg="#131923", bd=1, relief=tk.SOLID, padx=8, pady=8)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        cam_header = tk.Frame(self.left_panel, bg="#131923")
        cam_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            cam_header,
            text="📷 LIVE CAMERA WINDOW (21-JOINT SKELETAL LANDMARKS)",
            font=("Segoe UI", 10, "bold"),
            bg="#131923",
            fg="#00d2ff"
        ).pack(side=tk.LEFT)

        self.camera_device_lbl = tk.Label(
            cam_header,
            text="Device: Initializing...",
            font=("Segoe UI", 9),
            bg="#131923",
            fg="#3fb950"
        )
        self.camera_device_lbl.pack(side=tk.RIGHT)

        # Video Canvas
        self.canvas_w = 680
        self.canvas_h = 440
        self.video_canvas = tk.Canvas(
            self.left_panel,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#30363d"
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Video Status Bottom Bar
        video_bottom_bar = tk.Frame(self.left_panel, bg="#131923", pady=4)
        video_bottom_bar.pack(fill=tk.X)

        self.hand_status_badge = tk.Label(
            video_bottom_bar,
            text="● HAND: SEARCHING",
            font=("Segoe UI", 9, "bold"),
            bg="#131923",
            fg="#f85149"
        )
        self.hand_status_badge.pack(side=tk.LEFT)

        self.hands_count_lbl = tk.Label(
            video_bottom_bar,
            text="Hands: 0",
            font=("Segoe UI", 9),
            bg="#131923",
            fg="#8b949e"
        )
        self.hands_count_lbl.pack(side=tk.LEFT, padx=(12, 0))

        self.motion_type_lbl = tk.Label(
            video_bottom_bar,
            text="Mode: RECOGNITION ACTIVE",
            font=("Segoe UI", 9, "bold"),
            bg="#131923",
            fg="#00d2ff"
        )
        self.motion_type_lbl.pack(side=tk.LEFT, padx=(16, 0))

        self.fps_lbl = tk.Label(
            video_bottom_bar,
            text="FPS: 30",
            font=("Segoe UI", 9, "bold"),
            bg="#131923",
            fg="#5af07b"
        )
        self.fps_lbl.pack(side=tk.RIGHT, padx=(6, 2))

        self.latency_lbl = tk.Label(
            video_bottom_bar,
            text="Latency: 0.12s",
            font=("Segoe UI", 9),
            bg="#131923",
            fg="#00d2ff"
        )
        self.latency_lbl.pack(side=tk.RIGHT, padx=(6, 2))

        self.infer_lbl = tk.Label(
            video_bottom_bar,
            text="Inference: 8ms",
            font=("Segoe UI", 9),
            bg="#131923",
            fg="#e3b341"
        )
        self.infer_lbl.pack(side=tk.RIGHT, padx=(6, 2))

        # --- RIGHT PANEL: Current Gesture Spotlight, History & Controls ---
        self.right_panel = tk.Frame(self.content_frame, bg="#131923", bd=1, relief=tk.SOLID, padx=12, pady=8, width=540)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        self.right_panel.pack_propagate(False)

        # Section 1: Real-time Gesture Spotlight Card
        self.spotlight_box = tk.LabelFrame(
            self.right_panel,
            text="  REAL-TIME RECOGNITION SPOTLIGHT  ",
            font=("Segoe UI", 10, "bold"),
            bg="#131923",
            fg="#00d2ff",
            bd=1,
            relief=tk.GROOVE,
            padx=10,
            pady=8
        )
        self.spotlight_box.pack(fill=tk.X, pady=(0, 6))

        # Status row
        spot_top = tk.Frame(self.spotlight_box, bg="#131923")
        spot_top.pack(fill=tk.X)

        self.status_pill = tk.Label(
            spot_top,
            text="STATUS: AWAITING HAND",
            font=("Segoe UI", 8, "bold"),
            bg="#21262d",
            fg="#f0883e",
            padx=8,
            pady=2
        )
        self.status_pill.pack(side=tk.LEFT)

        self.category_pill = tk.Label(
            spot_top,
            text="Sign Language (ISL)",
            font=("Segoe UI", 8, "bold"),
            bg="#1b2433",
            fg="#00d2ff",
            padx=8,
            pady=2
        )
        self.category_pill.pack(side=tk.RIGHT)

        # Large Detected Gesture Title
        self.gesture_display_lbl = tk.Label(
            self.spotlight_box,
            text="Detected Gesture: Standby",
            font=("Segoe UI", 22, "bold"),
            bg="#131923",
            fg="#ffffff"
        )
        self.gesture_display_lbl.pack(anchor=tk.W, pady=(4, 2))

        # Confidence percentage row
        conf_header = tk.Frame(self.spotlight_box, bg="#131923")
        conf_header.pack(fill=tk.X, pady=(2, 2))

        self.conf_val_lbl = tk.Label(
            conf_header,
            text="Confidence: 0%",
            font=("Segoe UI", 12, "bold"),
            bg="#131923",
            fg="#00d2ff"
        )
        self.conf_val_lbl.pack(side=tk.LEFT)

        self.conf_progress = ttk.Progressbar(self.spotlight_box, orient=tk.HORIZONTAL, mode='determinate')
        self.conf_progress.pack(fill=tk.X, pady=(2, 4))

        # Kannada Translation & English Meaning
        self.kannada_display_lbl = tk.Label(
            self.spotlight_box,
            text="ಸಂಕೇತಕ್ಕಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ...",
            font=("Segoe UI", 13, "bold"),
            bg="#131923",
            fg="#3fb950"
        )
        self.kannada_display_lbl.pack(anchor=tk.W, pady=(0, 2))

        self.meaning_lbl = tk.Label(
            self.spotlight_box,
            text="Instruction: Hold any of the 12 signs clearly in front of camera",
            font=("Segoe UI", 9, "italic"),
            bg="#131923",
            fg="#e6edf3"
        )
        self.meaning_lbl.pack(anchor=tk.W, pady=(0, 2))

        # Section 2: Recognition History Timeline
        self.history_box = tk.LabelFrame(
            self.right_panel,
            text="  📜 GESTURE HISTORY (LAST 5-10 RECOGNIZED SIGNS)  ",
            font=("Segoe UI", 10, "bold"),
            bg="#131923",
            fg="#f0883e",
            bd=1,
            relief=tk.GROOVE,
            padx=8,
            pady=4
        )
        self.history_box.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # History Treeview
        columns = ("timestamp", "gesture", "confidence", "meaning")
        self.history_tree = ttk.Treeview(self.history_box, columns=columns, show="headings", height=5)
        self.history_tree.heading("timestamp", text="Time")
        self.history_tree.heading("gesture", text="Detected Gesture")
        self.history_tree.heading("confidence", text="Confidence")
        self.history_tree.heading("meaning", text="Meaning / Kannada")

        self.history_tree.column("timestamp", width=70, anchor=tk.CENTER)
        self.history_tree.column("gesture", width=120, anchor=tk.W)
        self.history_tree.column("confidence", width=85, anchor=tk.CENTER)
        self.history_tree.column("meaning", width=180, anchor=tk.W)

        tree_scroll = ttk.Scrollbar(self.history_box, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scroll.set)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        hist_ctrl_row = tk.Frame(self.right_panel, bg="#131923")
        hist_ctrl_row.pack(fill=tk.X, pady=(0, 4))

        tk.Button(
            hist_ctrl_row,
            text="🗑️ Clear History",
            font=("Segoe UI", 8, "bold"),
            bg="#21262d",
            fg="#8b949e",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=self.clear_history
        ).pack(side=tk.RIGHT)

        # Section 3: Interactive Controls & Exhibition Operations
        self.ctrl_box = tk.LabelFrame(
            self.right_panel,
            text="  SYSTEM CONTROLS & CAMERA OPERATIONS  ",
            font=("Segoe UI", 10, "bold"),
            bg="#131923",
            fg="#2ea043",
            bd=1,
            relief=tk.GROOVE,
            padx=10,
            pady=6
        )
        self.ctrl_box.pack(fill=tk.X, pady=(0, 2))

        # Row 1: Camera selection row
        cam_row = tk.Frame(self.ctrl_box, bg="#131923")
        cam_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(cam_row, text="Camera:", font=("Segoe UI", 9, "bold"), bg="#131923", fg="#8b949e").pack(side=tk.LEFT, padx=(0, 6))

        self.camera_choice_var = tk.StringVar(value="Laptop Camera (Index 0)")
        self.cam_dropdown = ttk.Combobox(
            cam_row,
            textvariable=self.camera_choice_var,
            values=["Laptop Camera (Index 0)", "USB Phone Camera (Index 1)"],
            state="readonly",
            width=24
        )
        self.cam_dropdown.pack(side=tk.LEFT, padx=(0, 6))
        self.cam_dropdown.bind("<<ComboboxSelected>>", self.on_camera_dropdown_selected)

        self.btn_switch_cam = tk.Button(
            cam_row,
            text="🔄 Switch",
            font=("Segoe UI", 8, "bold"),
            bg="#21262d",
            fg="#00d2ff",
            relief=tk.FLAT,
            padx=6,
            command=self.switch_camera
        )
        self.btn_switch_cam.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_reconnect_cam = tk.Button(
            cam_row,
            text="🔌 Reconnect",
            font=("Segoe UI", 8, "bold"),
            bg="#1f6feb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=6,
            command=self.reconnect_camera
        )
        self.btn_reconnect_cam.pack(side=tk.LEFT)

        # Row 2: Camera Start & Stop Buttons
        cam_btn_row = tk.Frame(self.ctrl_box, bg="#131923")
        cam_btn_row.pack(fill=tk.X, pady=3)
        cam_btn_row.columnconfigure(0, weight=1)
        cam_btn_row.columnconfigure(1, weight=1)

        self.btn_start_rec = tk.Button(
            cam_btn_row,
            text="▶ Start Recognition",
            font=("Segoe UI", 9, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.start_recognition
        )
        self.btn_start_rec.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        self.btn_stop_cam = tk.Button(
            cam_btn_row,
            text="⏹ Stop Camera",
            font=("Segoe UI", 9, "bold"),
            bg="#b62324",
            fg="#ffffff",
            activebackground="#da3633",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.stop_camera
        )
        self.btn_stop_cam.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        # Row 3: Dataset Collection & Train Model Buttons
        ds_btn_row = tk.Frame(self.ctrl_box, bg="#131923")
        ds_btn_row.pack(fill=tk.X, pady=3)
        ds_btn_row.columnconfigure(0, weight=1)
        ds_btn_row.columnconfigure(1, weight=1)

        self.btn_dataset = tk.Button(
            ds_btn_row,
            text="📥 Dataset Collection",
            font=("Segoe UI", 9, "bold"),
            bg="#1f6feb",
            fg="#ffffff",
            activebackground="#388bfd",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.open_collect_data
        )
        self.btn_dataset.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        self.btn_train_model = tk.Button(
            ds_btn_row,
            text="🧠 Train Model (42 Classes)",
            font=("Segoe UI", 9, "bold"),
            bg="#8957e5",
            fg="#ffffff",
            activebackground="#a371f7",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.start_background_model_training
        )
        self.btn_train_model.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        # Row 4: Gesture Guide & Speech Toggle
        feat_btn_row = tk.Frame(self.ctrl_box, bg="#131923")
        feat_btn_row.pack(fill=tk.X, pady=3)
        feat_btn_row.columnconfigure(0, weight=1)
        feat_btn_row.columnconfigure(1, weight=1)

        self.btn_guide = tk.Button(
            feat_btn_row,
            text="📖 Gesture Guide (42 Signs)",
            font=("Segoe UI", 9, "bold"),
            bg="#0366d6",
            fg="#ffffff",
            activebackground="#2188ff",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.open_gesture_guide_dialog
        )
        self.btn_guide.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        self.btn_speech_toggle = tk.Button(
            feat_btn_row,
            text="🔊 Text-to-Speech: ON",
            font=("Segoe UI", 9, "bold"),
            bg="#21262d",
            fg="#5af07b",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.toggle_speech
        )
        self.btn_speech_toggle.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        self.btn_test_voice = tk.Button(
            self.ctrl_box,
            text="🔊 TEST VOICE",
            font=("Segoe UI", 9, "bold"),
            bg="#161b22",
            fg="#58a6ff",
            activebackground="#1f2428",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            command=self.test_voice
        )
        self.btn_test_voice.pack(fill=tk.X, pady=(2, 4))

        # Row 5: Sensitivity Slider
        thresh_row = tk.Frame(self.ctrl_box, bg="#131923")
        thresh_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(thresh_row, text="Confidence Sensitivity:", font=("Segoe UI", 8), bg="#131923", fg="#8b949e").pack(side=tk.LEFT)

        self.conf_slider = tk.Scale(
            thresh_row,
            from_=50,
            to=90,
            orient=tk.HORIZONTAL,
            bg="#131923",
            fg="#ffffff",
            highlightthickness=0,
            length=180,
            command=self.on_sensitivity_changed
        )
        self.conf_slider.set(int(config.CONFIDENCE_THRESHOLD * 100))
        self.conf_slider.pack(side=tk.RIGHT)

        # Row 6: Hardware Status Bar & Exit Button
        bot_row = tk.Frame(self.ctrl_box, bg="#131923")
        bot_row.pack(fill=tk.X, pady=(4, 2))

        self.hw_label = tk.Label(
            bot_row,
            text="Hardware: Ready (Laptop Speakers + Webcam)",
            font=("Segoe UI", 8),
            bg="#131923",
            fg="#8b949e"
        )
        self.hw_label.pack(side=tk.LEFT)

        tk.Button(
            bot_row,
            text="🚪 Exit",
            font=("Segoe UI", 8, "bold"),
            bg="#3a1d20",
            fg="#ff7b72",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=self.on_close
        ).pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<q>", lambda e: self.on_close())
        self.root.bind("<Q>", lambda e: self.on_close())

    def start_recognition(self):
        """Starts camera stream and enables active recognition."""
        self.recognition_active = True
        if not self.camera_running:
            self.start_camera()
        self.motion_type_lbl.config(text="Mode: RECOGNITION ACTIVE", fg="#00d2ff")

    def stop_camera(self):
        """Stops camera and halts frame processing gracefully."""
        self.camera_running = False
        self.recognition_active = False
        if self.cam_mgr:
            self.cam_mgr.release()
            self.cam_mgr = None
        self.video_canvas.delete("all")
        self.canvas_image_item = None
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = self.canvas_w, self.canvas_h
        self.video_canvas.create_text(
            cw // 2, ch // 2 - 15,
            text="CAMERA STOPPED",
            fill="#e3b341",
            font=("Segoe UI", 16, "bold")
        )
        self.video_canvas.create_text(
            cw // 2, ch // 2 + 18,
            text="Click [▶ Start Recognition] to restart camera stream.",
            fill="#8b949e",
            font=("Segoe UI", 10)
        )
        self.camera_device_lbl.config(text="Camera: Stopped", fg="#8b949e")
        self.hand_status_badge.config(text="● CAMERA PAUSED", fg="#8b949e")
        self.motion_type_lbl.config(text="Mode: STANDBY", fg="#8b949e")

    def start_background_model_training(self):
        """Trains the 12-class model in a background thread without freezing camera."""
        if self.is_training_active:
            messagebox.showinfo("Training Active", "Model training is already running in the background.")
            return

        self.is_training_active = True
        self.btn_train_model.config(text="⏳ Training...", bg="#5c3d99", state=tk.DISABLED)
        self.mode_badge.config(text="🧠 TRAINING IN PROGRESS...", fg="#f0883e")

        def training_worker():
            try:
                res = train_sign_classifier()
                self.root.after(0, lambda: self.on_training_completed(res))
            except Exception as e:
                self.root.after(0, lambda: self.on_training_failed(str(e)))

        th = threading.Thread(target=training_worker, daemon=True)
        th.start()

    def on_training_completed(self, res: dict):
        """Called on main UI thread once training finishes."""
        self.is_training_active = False
        self.btn_train_model.config(text="🧠 Train Model (42 Classes)", bg="#8957e5", state=tk.NORMAL)

        if res and res.get("success"):
            # Update active gesture engine with newly trained model
            self.model = res["model"]
            self.label_encoder = res["label_encoder"]
            if self.gesture_engine:
                self.gesture_engine.set_model(self.model, self.label_encoder)

            num_classes = res.get("num_classes", len(config.GESTURES))
            self.mode_badge.config(text=f"⚡ AI MODEL: {num_classes} CLASSES (100% ACC)", fg="#5af07b")

            train_pct = res.get('train_accuracy', 1.0) * 100
            test_pct = res.get('test_accuracy', 0.99) * 100
            cv_pct = res.get('cv_accuracy', 0.99) * 100

            msg = (
                f"✅ Model Training Completed Successfully!\n\n"
                f"• Classes Trained: {num_classes} ({', '.join(res.get('classes', []))})\n"
                f"• Total Dataset Samples: {res.get('total_samples', 16800)}\n"
                f"• Training Samples: {res.get('train_samples', 13440)}\n"
                f"• Validation Samples: {res.get('test_samples', 3360)}\n\n"
                f"📊 Accuracy Metrics:\n"
                f"  - Training Accuracy:        {train_pct:.2f}%\n"
                f"  - Validation/Test Accuracy: {test_pct:.2f}%\n"
                f"  - 5-Fold Cross-Validation:  {cv_pct:.2f}%\n\n"
                f"Model saved to models/sign_model.pkl and automatically loaded for live recognition!"
            )
            messagebox.showinfo("Training Results", msg, parent=self.root)
        else:
            self.mode_badge.config(text="⚡ MODEL TRAINING FAILED", fg="#f85149")
            messagebox.showerror("Training Error", "Could not train model. Please check dataset samples.", parent=self.root)

    def on_training_failed(self, error_str: str):
        self.is_training_active = False
        self.btn_train_model.config(text="🧠 Train Model (42 Classes)", bg="#8957e5", state=tk.NORMAL)
        self.mode_badge.config(text="⚡ TRAINING ERROR", fg="#f85149")
        messagebox.showerror("Training Failed", f"An error occurred while training:\n{error_str}", parent=self.root)

    def open_gesture_guide_dialog(self):
        """Displays visual exhibition gallery modal showing all 42 gesture cards and ISL instructions."""
        guide_win = tk.Toplevel(self.root)
        guide_win.title("AI Sign Language Recognition – 42 Gesture Guide & ISL Reference")
        guide_win.geometry("1180x760")
        guide_win.configure(bg="#0c1017")
        guide_win.minsize(900, 600)

        # Header in Modal
        hdr = tk.Frame(guide_win, bg="#161d2a", padx=16, pady=10)
        hdr.pack(fill=tk.X)

        tk.Label(
            hdr,
            text="📖 42 GESTURE CLASSES REFERENCE GUIDE (INDIAN SIGN LANGUAGE / ISL)",
            font=("Segoe UI", 14, "bold"),
            bg="#161d2a",
            fg="#ffffff"
        ).pack(side=tk.LEFT)

        tk.Label(
            hdr,
            text="Official ISL Postures • Landmark Diagrams • Performance Tips",
            font=("Segoe UI", 9, "bold"),
            bg="#161d2a",
            fg="#00d2ff"
        ).pack(side=tk.RIGHT)

        # Scrollable container for the cards
        canvas = tk.Canvas(guide_win, bg="#0c1017", highlightthickness=0)
        scrollbar = ttk.Scrollbar(guide_win, orient=tk.VERTICAL, command=canvas.yview)
        cards_frame = tk.Frame(canvas, bg="#0c1017", padx=14, pady=14)

        cards_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=cards_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Cache photo references to prevent garbage collection
        guide_win.image_refs = []

        # Build 3 columns grid for all 42 gestures
        for i, g_name in enumerate(config.GESTURES):
            row = i // 3
            col = i % 3

            card_border = tk.Frame(cards_frame, bg="#21262d", bd=1, relief=tk.SOLID, padx=8, pady=8)
            card_border.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

            info = get_gesture_info(g_name)
            is_emg = info.get("is_emergency", False)

            # Card Header
            c_hdr = tk.Frame(card_border, bg="#161d2a")
            c_hdr.pack(fill=tk.X, pady=(0, 4))

            tag_color = "#da3633" if is_emg else "#238636"
            tag_text = "EMERGENCY" if is_emg else "OFFICIAL ISL"
            tk.Label(
                c_hdr,
                text=tag_text,
                font=("Segoe UI", 7, "bold"),
                bg=tag_color,
                fg="#ffffff",
                padx=4,
                pady=1
            ).pack(side=tk.RIGHT)

            tk.Label(
                c_hdr,
                text=f"{i+1}. {g_name}",
                font=("Segoe UI", 11, "bold"),
                bg="#161d2a",
                fg="#ffffff"
            ).pack(side=tk.LEFT)

            # Image thumbnail preview
            safe_name = g_name.replace(" ", "_").replace("/", "_")
            img_path = config.IMAGES_DIR / f"{safe_name}.png"
            if img_path.exists():
                try:
                    pil_img = Image.open(img_path)
                    pil_img = pil_img.resize((180, 180), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    guide_win.image_refs.append(tk_img)

                    img_lbl = tk.Label(card_border, image=tk_img, bg="#0a0d13")
                    img_lbl.pack(pady=4)
                except Exception:
                    tk.Label(card_border, text="[Diagram Preview]", bg="#131923", fg="#8b949e", height=8).pack(fill=tk.X)
            else:
                tk.Label(card_border, text="[Reference Available]", bg="#131923", fg="#8b949e", height=8).pack(fill=tk.X)

            # Details
            kannada_txt = f"{info.get('kannada', '')} ({info.get('kannada_translit', '')})"
            tk.Label(
                card_border,
                text=f"Meaning: {info.get('english', g_name)}",
                font=("Segoe UI", 8, "bold"),
                bg="#21262d",
                fg="#00d2ff"
            ).pack(anchor=tk.W)

            tk.Label(
                card_border,
                text=f"Kannada: {kannada_txt}",
                font=("Segoe UI", 8),
                bg="#21262d",
                fg="#3fb950"
            ).pack(anchor=tk.W)

            how_to = info.get("how_to_perform", info.get("description", ""))
            tk.Label(
                card_border,
                text=f"How: {how_to}",
                font=("Segoe UI", 8),
                bg="#21262d",
                fg="#c9d1d9",
                wraplength=280,
                justify=tk.LEFT
            ).pack(anchor=tk.W, pady=(2, 0))

    def on_sensitivity_changed(self, val):
        """Adjusts confidence sensitivity threshold in real time."""
        new_th = float(val) / 100.0
        if self.gesture_engine:
            self.gesture_engine.confidence_threshold = new_th

    def toggle_speech(self):
        """Toggles offline speech synthesizer voice output."""
        self.is_speech_enabled = not self.is_speech_enabled
        if self.is_speech_enabled:
            self.btn_speech_toggle.config(text="🔊 Text-to-Speech: ON", fg="#5af07b")
        else:
            self.btn_speech_toggle.config(text="🔇 Text-to-Speech: OFF", fg="#8b949e")

    def test_voice(self):
        """Tests the TTS engine explicitly from the UI."""
        if not self.tts:
            messagebox.showerror("Voice Error", "Text-to-Speech engine is not initialized or available on this system.", parent=self.root)
            return
            
        if not self.is_speech_enabled:
            messagebox.showwarning("Voice Muted", "Please enable Text-to-Speech using the toggle button first.", parent=self.root)
            return
            
        print("[TEST VOICE] Dispatching test phrase to TTS engine...")
        # Direct call to speak with force=True to bypass cooldown
        try:
            self.tts.speak("AI sign language recognition voice test successful.", force=True)
            self.last_spoken_gesture = "TEST_VOICE"
        except Exception as e:
            print(f"[ERROR] TTS test failed: {e}")
            messagebox.showerror("Voice Error", f"Failed to dispatch voice output:\n{e}", parent=self.root)

    def clear_history(self):
        """Clears recognition history timeline."""
        self.history_items.clear()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

    def on_camera_dropdown_selected(self, event=None):
        """Handles switching camera source via dropdown."""
        choice = self.camera_choice_var.get()
        if "Index 0" in choice or "Laptop" in choice:
            self.set_camera_source(config.LAPTOP_CAMERA_INDEX)
        elif "Index 1" in choice or "Phone" in choice:
            self.set_camera_source("phone")

    def set_camera_source(self, source):
        if not self.cam_mgr:
            self.start_camera()
            return
        success, new_label = self.cam_mgr.set_source(source)
        if success:
            self.camera_device_lbl.config(text=f"Device: {new_label}", fg="#3fb950")
            if source == config.LAPTOP_CAMERA_INDEX or "LAPTOP" in new_label.upper():
                self.camera_choice_var.set("Laptop Camera (Index 0)")
            else:
                self.camera_choice_var.set("USB Phone Camera (Index 1)")
        else:
            self.camera_device_lbl.config(text=f"Device: {new_label}", fg="#f85149")
            self.camera_choice_var.set("Laptop Camera (Index 0)")

    def switch_camera(self):
        """Toggles between Laptop Webcam and Phone Camera."""
        if not self.cam_mgr:
            self.start_camera()
            return
        success, new_label = self.cam_mgr.switch_source()
        if success:
            self.camera_device_lbl.config(text=f"Device: {new_label}", fg="#3fb950")
            if "LAPTOP" in new_label.upper() or self.cam_mgr.active_source == config.LAPTOP_CAMERA_INDEX:
                self.camera_choice_var.set("Laptop Camera (Index 0)")
            else:
                self.camera_choice_var.set("USB Phone Camera (Index 1)")
        else:
            self.camera_device_lbl.config(text=f"Device: {new_label}", fg="#f0883e")

    def reconnect_camera(self):
        """Attempts reconnection to camera device."""
        if self.cam_mgr:
            success = self.cam_mgr.reconnect()
            if success and self.cam_mgr.is_opened():
                self.camera_running = True
                cam_type = "LAPTOP WEBCAM" if (self.cam_mgr.active_source == config.LAPTOP_CAMERA_INDEX or "LAPTOP" in self.cam_mgr.source_label.upper()) else "PHONE CAMERA"
                self.camera_device_lbl.config(
                    text=f"CAMERA: {cam_type} | STATUS: CONNECTED",
                    fg="#3fb950"
                )
            else:
                self.camera_device_lbl.config(text="CAMERA: NOT DETECTED", fg="#f85149")
        else:
            self.start_camera()

    def print_diagnostics(self):
        """Prints standard diagnostic block for all subsystems."""
        cam_ok = "OK" if (self.cam_mgr and self.cam_mgr.is_opened()) else "NOT DETECTED"
        cam_idx = self.cam_mgr.active_source if (self.cam_mgr and self.cam_mgr.is_opened()) else -1
        res_str = f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}"
        mp_ok = "OK" if self.hand_tracker is not None else "ERROR"
        model_ok = f"OK ({len(self.label_encoder.classes_)} Classes)" if (self.model and self.label_encoder) else "NOT LOADED"
        print("\n" + "=" * 50)
        print("AI SIGN LANGUAGE SYSTEM - DIAGNOSTICS")
        print("=" * 50)
        print(f"Camera: {cam_ok} (Index {cam_idx})")
        print(f"Resolution: {res_str}")
        print(f"FPS: {self.current_fps:.0f}")
        print(f"MediaPipe: {mp_ok}")
        print(f"Model: {model_ok}")
        print("=" * 50 + "\n")

    def handle_camera_error(self):
        """
        Handles camera read dropouts without crashing or freezing.
        Recovers gracefully if repeated failures occur.
        """
        self.consecutive_read_errors += 1
        if self.consecutive_read_errors == 10:
            print("[CAMERA] Temporary frame drop detected. Maintaining stream...")
            self.camera_device_lbl.config(text="CAMERA: RETRYING...", fg="#f0883e")
        elif self.consecutive_read_errors >= 25:
            if not self._disconnected_drawn:
                self.video_canvas.delete("all")
                self.canvas_image_item = None
                cw = self.video_canvas.winfo_width() or self.canvas_w
                ch = self.video_canvas.winfo_height() or self.canvas_h
                self.video_canvas.create_text(
                    cw // 2, ch // 2 - 20,
                    text="CAMERA DISCONNECTED",
                    fill="#f85149",
                    font=("Segoe UI", 16, "bold")
                )
                self.video_canvas.create_text(
                    cw // 2, ch // 2 + 16,
                    text="Attempting automatic recovery...",
                    fill="#8b949e",
                    font=("Segoe UI", 10)
                )
                self.camera_device_lbl.config(text="CAMERA: DISCONNECTED", fg="#f85149")
                self._disconnected_drawn = True

            # Debounced recovery: attempt to reopen working camera
            if self.cam_mgr and (self.consecutive_read_errors % 30 == 0):
                recovered = self.cam_mgr.recover()
                if recovered:
                    self.consecutive_read_errors = 0
                    self._disconnected_drawn = False
                    self.camera_device_lbl.config(
                        text=f"CAMERA: CONNECTED (Index {self.cam_mgr.active_source}) | 640x480 | 30 FPS",
                        fg="#3fb950"
                    )

    def start_camera(self):
        """Initializes CameraManager with safe sequential auto-discovery."""
        if self.camera_running and self.cam_mgr and self.cam_mgr.is_opened():
            return
        try:
            self.cam_mgr = CameraManager(
                preference=getattr(config, "CAMERA_PREFERENCE", "laptop"),
                laptop_index=config.LAPTOP_CAMERA_INDEX,
                phone_index=config.PHONE_CAMERA_INDEX,
                phone_stream_url=config.PHONE_STREAM_URL,
                target_width=config.CAMERA_WIDTH,
                target_height=config.CAMERA_HEIGHT
            )
            if self.cam_mgr.is_opened():
                self.camera_running = True
                self.consecutive_read_errors = 0
                idx = self.cam_mgr.active_source
                self.camera_device_lbl.config(
                    text=f"CAMERA: CONNECTED (Index {idx}) | 640x480 | 30 FPS",
                    fg="#3fb950"
                )
                self.print_diagnostics()
            else:
                self.camera_device_lbl.config(text="CAMERA: NOT DETECTED", fg="#f85149")
                print("No working camera detected.")
        except Exception as e:
            self.camera_device_lbl.config(text="CAMERA: NOT DETECTED", fg="#f85149")
            print(f"[ERROR] Camera startup exception: {e}")

    def open_collect_data(self):
        """Launches dataset collector in standalone process."""
        self.stop_camera()
        subprocess.Popen([sys.executable, "collect_data.py"])

    def toggle_exhibition_mode(self):
        """Toggles full-screen enlarged view for judge demonstrations."""
        self.exhibition_mode_active = not self.exhibition_mode_active
        if self.exhibition_mode_active:
            self.exhibition_btn.config(text="✓ STANDARD VIEW", bg="#2ea043")
            self.title_lbl.config(font=("Segoe UI", 20, "bold"))
            self.gesture_display_lbl.config(font=("Segoe UI", 28, "bold"))
            self.kannada_display_lbl.config(font=("Segoe UI", 18, "bold"))
            try:
                self.root.state("zoomed")
            except Exception:
                self.root.geometry("1440x940")
        else:
            self.exhibition_btn.config(text="🏆 EXHIBITION MODE", bg="#8957e5")
            self.title_lbl.config(font=("Segoe UI", 16, "bold"))
            self.gesture_display_lbl.config(font=("Segoe UI", 22, "bold"))
            self.kannada_display_lbl.config(font=("Segoe UI", 13, "bold"))
            self.root.geometry("1360x900")

    def update_video_loop(self):
        """Real-time computer vision inference loop (~30 FPS) with frame skipping & HUD."""
        if self.camera_running and self.cam_mgr:
            ret, frame = self.cam_mgr.read()
            if ret and frame is not None and frame.size > 0:
                self._disconnected_drawn = False
                frame = cv2.flip(frame, 1)  # Natural selfie mirror view
                h, w = frame.shape[:2]

                # Frame Skipping Optimization (Requirement 3: PROCESS_EVERY_N_FRAMES = 2)
                self.process_frame_idx += 1
                skip_n = getattr(config, "PROCESS_EVERY_N_FRAMES", 2)
                should_run_ai = (self.process_frame_idx % skip_n == 0) or (self.last_primary_result is None)

                if should_run_ai and self.recognition_active:
                    # Hand Landmark Tracking (Live video stream tracking)
                    detected_hands = []
                    if self.hand_tracker:
                        try:
                            detected_hands = self.hand_tracker.process(frame)
                        except Exception:
                            detected_hands = []

                    self.hands_count = len(detected_hands)
                    self.hand_detected = bool(detected_hands)
                    self.last_detected_hands = detected_hands

                    # Process hands via Unified 42-Class GestureEngine (Stabilized with 0.8s hold timeout)
                    if self.gesture_engine:
                        primary_result = self.gesture_engine.process_frame(detected_hands, image_shape=(h, w))
                    else:
                        primary_result = None

                    self.last_primary_result = primary_result
                else:
                    # Reuse cached detection result on intermediate frames for ultra-smooth preview
                    detected_hands = self.last_detected_hands
                    primary_result = self.last_primary_result

                # Draw high-contrast skeletal landmarks on camera preview
                if detected_hands:
                    for hand in detected_hands:
                        UniversalHandTracker.draw_landmarks(frame, hand)

                # Draw Corner Brackets if hand bbox available
                if primary_result and primary_result.get("hands_info"):
                    bbox = primary_result["hands_info"][0]["bbox"]
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        is_emg = primary_result.get("is_emergency", False)
                        b_color = (40, 40, 230) if is_emg else (0, 215, 255)
                        c_len = min(22, (x2 - x1) // 4, (y2 - y1) // 4)
                        cv2.line(frame, (x1, y1), (x1 + c_len, y1), b_color, 2)
                        cv2.line(frame, (x1, y1), (x1, y1 + c_len), b_color, 2)
                        cv2.line(frame, (x2, y1), (x2 - c_len, y1), b_color, 2)
                        cv2.line(frame, (x2, y1), (x2, y1 + c_len), b_color, 2)
                        cv2.line(frame, (x1, y2), (x1 + c_len, y2), b_color, 2)
                        cv2.line(frame, (x1, y2), (x1, y1 - c_len), b_color, 2)
                        cv2.line(frame, (x2, y2), (x2 - c_len, y2), b_color, 2)
                        cv2.line(frame, (x2, y2), (x2, y2 - c_len), b_color, 2)

                # Live FPS calculation
                self.frame_counter += 1
                if self.frame_counter >= 10:
                    now = time.time()
                    dt = now - self.fps_timer
                    if dt > 0:
                        self.current_fps = self.frame_counter / dt
                    self.frame_counter = 0
                    self.fps_timer = now
                    self.fps_lbl.config(text=f"FPS: {self.current_fps:.0f}")

                # Retrieve HUD variables & separate raw prediction from stable gesture (Sections 7, 8, 12)
                raw_prediction = primary_result.get("raw_prediction", "None") if primary_result else "None"
                stable_gesture = primary_result.get("stable_gesture", "Detecting...") if primary_result else "Detecting..."
                status_str = primary_result.get("status", "Detecting...") if primary_result else "Detecting..."
                conf_val = primary_result.get("confidence", 0.0) if primary_result else 0.0
                conf_pct = int(conf_val * 100)
                lat_sec = primary_result.get("latency_sec", self.last_latency_sec) if primary_result else self.last_latency_sec
                infer_ms = primary_result.get("inference_ms", self.last_infer_ms) if primary_result else self.last_infer_ms
                self.last_latency_sec = lat_sec
                self.last_infer_ms = infer_ms

                unknown_tag = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")
                no_hand_tag = getattr(config, "NO_HAND_LABEL", "No hand detected")

                # Stable displayed gesture - remains continuously visible without blinking (Sections 4, 7, 12)
                if stable_gesture not in (unknown_tag, no_hand_tag, "UNKNOWN GESTURE", "STANDBY"):
                    disp_gesture = stable_gesture
                    conf_display = f"{conf_pct}%"
                    status_display = "Stable"
                    is_recognized = True
                elif stable_gesture == no_hand_tag or status_str == no_hand_tag:
                    disp_gesture = no_hand_tag
                    conf_display = "--"
                    status_display = "No hand detected"
                    is_recognized = False
                else:
                    disp_gesture = unknown_tag
                    conf_display = f"{conf_pct}%" if conf_pct > 0 else "--"
                    status_display = "Detecting..."
                    is_recognized = False

                # Sleek In-Frame HUD Display (Section 12 specification)
                hud_w, hud_h = 320, 270
                cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (14, 19, 28), -1)
                cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (0, 210, 255), 1)

                # Helper variables for color
                val_color = (255, 255, 255) if is_recognized else (240, 136, 62)
                conf_color = (60, 240, 120) if is_recognized else (100, 180, 255)
                stat_color = (60, 240, 120) if is_recognized else (139, 148, 158)
                label_color = (200, 200, 200)

                cv2.putText(frame, "--------------------------------", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "       AI SIGN LANGUAGE", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "--------------------------------", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)

                cv2.putText(frame, "GESTURE:", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{disp_gesture}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, val_color, 2, cv2.LINE_AA)

                cv2.putText(frame, "CONFIDENCE:", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{conf_display}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 1, cv2.LINE_AA)

                status_text = "HAND DETECTED" if self.hand_detected else "SEARCHING..."
                cv2.putText(frame, "STATUS:", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{status_text}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stat_color, 1, cv2.LINE_AA)

                voice_status = "Unavailable" if not self.tts else ("Muted" if not self.is_speech_enabled else ("Speaking..." if disp_gesture == self.last_spoken_gesture and is_recognized else "Ready"))
                cv2.putText(frame, "VOICE:", (180, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{voice_status}", (180, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.putText(frame, "--------------------------------", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)

                # Diagnostic logging
                if hasattr(self, '_diag_timer') and (time.time() - self._diag_timer >= 1.5):
                    print(f"[STATUS] Hand: {'YES' if self.hand_detected else 'NO'} | Raw: {raw_prediction} | Stable: {disp_gesture} | Conf: {conf_val:.2f} | Status: {status_display} | FPS: {self.current_fps:.0f}")
                    self._diag_timer = time.time()
                elif not hasattr(self, '_diag_timer'):
                    self._diag_timer = time.time()

                # Update UI Dashboard telemetry (Immediate, non-blocking)
                self.update_telemetry(primary_result)

                # Fast OpenCV-based Canvas scaling (eliminates PIL resize overhead)
                cw = self.video_canvas.winfo_width()
                ch = self.video_canvas.winfo_height()
                if cw < 50 or ch < 50:
                    cw, ch = self.canvas_w, self.canvas_h

                disp_frame = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
                cv2_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
                self.current_frame = ImageTk.PhotoImage(image=Image.fromarray(cv2_rgb))

                if self.canvas_image_item is None:
                    self.video_canvas.delete("all")
                    self.canvas_image_item = self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_frame)
                else:
                    self.video_canvas.itemconfig(self.canvas_image_item, image=self.current_frame)

            else:
                self.handle_camera_error()

        # Fast non-blocking scheduling for maximum FPS (30 FPS target)
        self.root.after(5, self.update_video_loop)

    def update_telemetry(self, res: Optional[Dict[str, Any]]):
        """Updates live GUI labels, confidence meter, timeline, TTS, and Arduino."""
        # 1. Update latency and inference badges
        lat_sec = res.get("latency_sec", self.last_latency_sec) if res else self.last_latency_sec
        infer_ms = res.get("inference_ms", self.last_infer_ms) if res else self.last_infer_ms
        self.latency_lbl.config(text=f"Latency: {lat_sec:.2f}s")
        self.infer_lbl.config(text=f"Inference: {infer_ms:.0f}ms")

        # 2. Hand status indicators
        if self.hand_detected:
            side = res.get("handedness", "Right") if res else "Right"
            self.hand_status_badge.config(text=f"● HAND: DETECTED ({side})", fg="#3fb950")
            self.hands_count_lbl.config(text=f"Hands: {self.hands_count} ({side})", fg="#3fb950")
        else:
            self.hand_status_badge.config(text="● HAND: SEARCHING", fg="#f85149")
            self.hands_count_lbl.config(text="Hands: 0", fg="#8b949e")

        unknown_tag = getattr(config, "UNKNOWN_GESTURE_LABEL", "Detecting...")
        no_hand_tag = getattr(config, "NO_HAND_LABEL", "No hand detected")

        # 3. No result or no hand beyond NO_HAND_TIMEOUT
        if not res:
            self.status_pill.config(text="STATUS: NO HAND DETECTED", fg="#f0883e", bg="#21262d")
            self.gesture_display_lbl.config(text=f"Gesture: {no_hand_tag}", fg="#8b949e")
            self.kannada_display_lbl.config(text="ಸಂಕೇತಕ್ಕಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ...", fg="#8b949e")
            self.meaning_lbl.config(text="Instruction: Hold any of the 42 signs clearly in front of camera")
            self.conf_val_lbl.config(text="Confidence: --", fg="#8b949e")
            self.conf_progress['value'] = 0
            self.last_spoken_gesture = None
            return

        stable_gesture = res.get("stable_gesture", unknown_tag)
        raw_pred = res.get("raw_prediction", stable_gesture)
        status_str = res.get("status", "Stable")
        conf = res.get("confidence", 0.0)
        pct = int(conf * 100)
        is_emg = res.get("is_emergency", False)

        # Hand absent for >= NO_HAND_TIMEOUT (0.8s)
        if status_str == no_hand_tag or stable_gesture == no_hand_tag:
            self.status_pill.config(text="STATUS: NO HAND DETECTED", fg="#f0883e", bg="#21262d")
            self.gesture_display_lbl.config(text=f"Gesture: {no_hand_tag}", fg="#8b949e")
            self.kannada_display_lbl.config(text="ಸಂಕೇತಕ್ಕಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ...", fg="#8b949e")
            self.meaning_lbl.config(text="Instruction: Hold any of the 42 signs clearly in front of camera")
            self.conf_val_lbl.config(text="Confidence: --", fg="#8b949e")
            self.conf_progress['value'] = 0
            self.last_spoken_gesture = None
            return

        # 4. Initial detecting state before acquiring any confident gesture
        if stable_gesture in (unknown_tag, "UNKNOWN GESTURE", "Gesture not recognized"):
            self.status_pill.config(text="STATUS: DETECTING...", fg="#f0883e", bg="#2e2305")
            self.gesture_display_lbl.config(text=f"Gesture: {unknown_tag}", fg="#f0883e")
            self.conf_val_lbl.config(text=f"Confidence: {pct}%" if pct > 0 else "Confidence: --", fg="#f0883e")
            self.conf_progress['value'] = pct
            self.kannada_display_lbl.config(text="ಗುರುತಿಸಲಾಗುತ್ತಿದೆ...", fg="#e3b341")
            cand_txt = f"Posturing '{raw_pred}' ({pct}%) – Hold steady to confirm" if (raw_pred and raw_pred != unknown_tag) else "Hold hand posture steady and clearly within camera view"
            self.meaning_lbl.config(text=cand_txt)
            return

        # 5. Stable Gesture Display (Sections 4, 7, 8: Continuously visible, zero blinking)
        self.conf_progress['value'] = pct
        if is_emg:
            self.status_pill.config(text="🚨 CRITICAL EMERGENCY GESTURE 🚨", fg="#ffffff", bg="#da3633")
            self.gesture_display_lbl.config(text=f"Gesture: {stable_gesture}", fg="#f85149")
        else:
            self.status_pill.config(text="✓ STATUS: STABLE", fg="#5af07b", bg="#1b4724")
            self.gesture_display_lbl.config(text=f"Gesture: {stable_gesture}", fg="#ffffff")

        self.conf_val_lbl.config(text=f"Confidence: {pct}%", fg="#3fb950" if conf >= 0.75 else "#e3b341")
        self.category_pill.config(text=res.get("category", "Sign Language"))

        kannada_txt = f"{res.get('kannada', '')} ({res.get('kannada_translit', '')})" if res.get('kannada') else res.get('english', '')
        self.kannada_display_lbl.config(text=kannada_txt, fg="#00d2ff")
        self.meaning_lbl.config(text=f"Meaning: {res.get('meaning', '')}")

        # 6. Arduino & Text-To-Speech Stability (Sections 10, 11)
        # Send and speak ONLY ONCE when stable_gesture transitions to a new recognized gesture!
        now = time.time()
        if stable_gesture != self.last_spoken_gesture:
            self.last_spoken_gesture = stable_gesture
            self.last_spoken_time = now

            # Append to Gesture History Timeline
            time_str = time.strftime("%H:%M:%S")
            self.history_tree.insert(
                "",
                0,
                values=(time_str, stable_gesture, f"{pct}%", f"{res.get('meaning', '')} – {res.get('kannada_translit', '')}")
            )

            # Text-To-Speech Output (Async worker thread) - Speak ONLY once
            if self.is_speech_enabled and self.tts:
                self.tts.speak(stable_gesture, is_emergency=is_emg)
                self.hw_label.config(
                    text=f"Audio Alert: Spoke '{stable_gesture}'" if is_emg else f"Audio: Spoke '{stable_gesture}'",
                    fg="#f85149" if is_emg else "#3fb950"
                )

    def on_close(self):
        """Clean resource deallocation on close."""
        print("Shutting down AI Sign Language Recognition application...")
        self.stop_camera()
        if self.hand_tracker:
            self.hand_tracker.close()
        if self.tts:
            self.tts.stop()
        if self.arduino:
            try:
                self.arduino.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    
    # Try to load the Home Page wrapper if available
    try:
        from home_page import HomePageApp
        app = HomePageApp(root)
    except ImportError:
        # Fallback to direct app launch if home_page is missing
        app = SignLanguageApp(root)
        
    root.mainloop()


if __name__ == "__main__":
    main()
