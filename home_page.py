import tkinter as tk
from tkinter import messagebox
import math
import time

class HomePageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI SIGN LANGUAGE RECOGNITION - HOME")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0c1017")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.animating = True
        self.angle = 0.0
        self.build_ui()
        self.animate()

    def build_ui(self):
        main_frame = tk.Frame(self.root, bg="#0c1017")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Left side: Text & Buttons
        left_panel = tk.Frame(main_frame, bg="#0c1017")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right side: 3D Canvas
        right_panel = tk.Frame(main_frame, bg="#0c1017")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_panel, text="------------------------------------------------", font=("Courier", 12), bg="#0c1017", fg="#00d2ff").pack(anchor=tk.W, pady=(0,10))
        
        tk.Label(left_panel, text="AI SIGN LANGUAGE\nRECOGNITION", font=("Segoe UI", 36, "bold"), bg="#0c1017", fg="#ffffff", justify=tk.LEFT).pack(anchor=tk.W)
        
        tk.Label(left_panel, text="Breaking Communication Barriers with Artificial Intelligence", font=("Segoe UI", 14, "italic"), bg="#0c1017", fg="#8b949e").pack(anchor=tk.W, pady=(10, 30))
        
        tk.Label(left_panel, text="✦ 42 GESTURE CLASSES\n✦ REAL-TIME AI\n✦ STABLE VOICE OUTPUT\n✦ REAL-TIME FEEDBACK", font=("Segoe UI", 12, "bold"), bg="#0c1017", fg="#3fb950", justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 40))
        
        # Buttons
        btn_style = {"font": ("Segoe UI", 12, "bold"), "fg": "#ffffff", "relief": tk.FLAT, "padx": 20, "pady": 10, "width": 25}
        
        btn_start = tk.Button(left_panel, text="▶ START RECOGNITION", bg="#238636", activebackground="#2ea043", command=self.start_recognition, **btn_style)
        btn_start.pack(anchor=tk.W, pady=8)
        
        btn_guide = tk.Button(left_panel, text="📖 GESTURE GUIDE", bg="#1f6feb", activebackground="#388bfd", command=self.open_guide, **btn_style)
        btn_guide.pack(anchor=tk.W, pady=8)
        
        btn_about = tk.Button(left_panel, text="ℹ ABOUT PROJECT", bg="#8957e5", activebackground="#a371f7", command=self.open_about, **btn_style)
        btn_about.pack(anchor=tk.W, pady=8)
        
        tk.Label(left_panel, text="------------------------------------------------", font=("Courier", 12), bg="#0c1017", fg="#00d2ff").pack(anchor=tk.W, pady=(20,0))
        
        # 3D Canvas
        self.canvas = tk.Canvas(right_panel, bg="#0c1017", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 3D hand node definitions (simple skeleton for a hand)
        # x, y, z relative to center
        self.base_nodes = [
            (0, 100, 0),      # 0: Wrist
            (-40, 20, 20),    # 1: Thumb base
            (-60, -20, 30),   # 2: Thumb mid
            (-70, -60, 40),   # 3: Thumb tip
            (-20, 0, 0),      # 4: Index base
            (-30, -50, 0),    # 5: Index mid
            (-40, -100, 0),   # 6: Index tip
            (0, -10, 0),      # 7: Middle base
            (0, -70, 0),      # 8: Middle mid
            (0, -120, 0),     # 9: Middle tip
            (20, -5, 0),      # 10: Ring base
            (30, -60, 0),     # 11: Ring mid
            (40, -105, 0),    # 12: Ring tip
            (40, 10, -10),    # 13: Pinky base
            (50, -40, -15),   # 14: Pinky mid
            (60, -80, -20),   # 15: Pinky tip
        ]
        
        self.edges = [
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6),
            (0,7), (7,8), (8,9),
            (0,10), (10,11), (11,12),
            (0,13), (13,14), (14,15),
            (4,7), (7,10), (10,13) # Palm connections
        ]

    def animate(self):
        if not self.animating:
            return
            
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 50: width = 500
        if height < 50: height = 600
        
        cx, cy = width / 2, height / 2
        
        # Rotation matrices
        self.angle += 0.02
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        
        # Subtle float effect
        float_y = math.sin(time.time() * 2) * 15
        
        projected = []
        for x, y, z in self.base_nodes:
            # Scale
            s = 1.5
            x, y, z = x*s, y*s, z*s
            
            # Rotate around Y axis
            rx = x * cos_a - z * sin_a
            rz = x * sin_a + z * cos_a
            ry = y
            
            # Rotate slightly around X axis for better view
            tilt = 0.3
            ry2 = ry * math.cos(tilt) - rz * math.sin(tilt)
            rz2 = ry * math.sin(tilt) + rz * math.cos(tilt)
            
            # Perspective projection
            fov = 400
            z_offset = 300
            factor = fov / (fov + rz2 + z_offset)
            
            px = rx * factor + cx
            py = ry2 * factor + cy + float_y
            
            projected.append((px, py, factor))
            
        # Draw edges
        for i, j in self.edges:
            p1 = projected[i]
            p2 = projected[j]
            # Color intensity based on depth (factor)
            intensity = int(min(255, max(50, 255 * (p1[2] + p2[2])/2)))
            color = f"#{0:02x}{int(intensity*0.8):02x}{intensity:02x}"
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=2)
            
        # Draw nodes with glow
        for px, py, factor in projected:
            r = 4 * factor
            intensity = int(min(255, max(50, 255 * factor)))
            glow_color = f"#{0:02x}{int(intensity*0.5):02x}{int(intensity*0.7):02x}"
            core_color = "#ffffff"
            
            self.canvas.create_oval(px-r-2, py-r-2, px+r+2, py+r+2, fill=glow_color, outline="")
            self.canvas.create_oval(px-r, py-r, px+r, py+r, fill=core_color, outline="")
            
        self.root.after(30, self.animate)

    def start_recognition(self):
        self.animating = False
        self.root.withdraw()
        
        import main
        top = tk.Toplevel(self.root)
        app = main.SignLanguageApp(top)
        
        # Override the on_close to return to home page
        original_close = app.on_close
        
        def custom_close():
            # Stop camera and threads safely
            print("Returning to Home Page...")
            app.stop_camera()
            if app.hand_tracker: app.hand_tracker.close()
            if app.tts: app.tts.stop()
            if getattr(app, 'arduino', None):
                try: app.arduino.close()
                except Exception: pass
            top.destroy()
            
            # Resume home page
            self.root.deiconify()
            self.animating = True
            self.animate()
            
        top.protocol("WM_DELETE_WINDOW", custom_close)
        app.on_close = custom_close

    def open_guide(self):
        # Temporarily use SignLanguageApp's gesture guide logic by creating a dummy one or just displaying simple list
        import config
        from utils.gesture_database import get_gesture_info
        
        guide_win = tk.Toplevel(self.root)
        guide_win.title("AI Sign Language Recognition - Gesture Guide")
        guide_win.geometry("900x600")
        guide_win.configure(bg="#0c1017")
        
        tk.Label(guide_win, text="42 GESTURE CLASSES REFERENCE", font=("Segoe UI", 16, "bold"), bg="#0c1017", fg="#ffffff").pack(pady=15)
        
        canvas = tk.Canvas(guide_win, bg="#0c1017", highlightthickness=0)
        scrollbar = tk.Scrollbar(guide_win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0c1017")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for i, g_name in enumerate(config.GESTURES):
            info = get_gesture_info(g_name)
            is_emg = info.get("is_emergency", False)
            
            row = i // 3
            col = i % 3
            
            card = tk.Frame(scrollable_frame, bg="#21262d", bd=1, relief=tk.SOLID, padx=10, pady=10)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            tk.Label(card, text=f"{i+1}. {g_name}", font=("Segoe UI", 12, "bold"), bg="#21262d", fg="#ffffff" if not is_emg else "#f85149").pack(anchor=tk.W)
            tk.Label(card, text=f"Meaning: {info.get('english', g_name)}", font=("Segoe UI", 9), bg="#21262d", fg="#00d2ff").pack(anchor=tk.W)
            if is_emg:
                tk.Label(card, text="EMERGENCY", font=("Segoe UI", 8, "bold"), bg="#da3633", fg="#ffffff").pack(anchor=tk.W, pady=(5,0))

    def open_about(self):
        about_text = """
AI SIGN LANGUAGE RECOGNITION PROJECT

A real-time edge AI system designed to bridge the communication gap.

TECHNOLOGY STACK:
• Webcam & OpenCV for real-time video streaming
• MediaPipe for high-precision 21-point hand landmark tracking
• Machine Learning (Random Forest) for 42-class gesture classification
• Text-to-Speech (pyttsx3) for stable, non-blocking voice output
• Real-time Audio & Visual feedback with emergency alerts

FEATURES:
• Real-time classification at 30+ FPS
• Stability engine prevents noise and repeated TTS spam
• 42 standardized gestures mapped natively to phrases

Built to run entirely offline on Windows using standard laptop cameras.
"""
    def on_close(self):
        self.animating = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HomePageApp(root)
    root.mainloop()
