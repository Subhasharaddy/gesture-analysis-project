# Android Phone Camera as USB Webcam: Complete Integration Guide
### *High-Definition, Zero-Latency Vision Input for AI Sign Language Recognition*

---

## 📌 Architecture Overview

```text
  📱 Android Phone Camera (1080p / 60 FPS)
         │
         │  ⚡ High-Speed USB Cable (USB-C / USB 3.0)
         ▼
  💻 Windows Laptop
         │
         ├──► Method A: Native Android 14+ USB UVC Webcam  ──┐
         ├──► Method B: DroidCam USB ADB (Zero Wi-Fi lag)   ──┼──► Windows DirectShow Device
         └──► Method C: Iriun 4K USB DirectShow             ──┘   (or Localhost Port 4747)
                                                                             │
                                                                             ▼
                                                                     cv2.VideoCapture()
                                                                             │
                                                                             ▼
                                                                   MediaPipe Hands (21 Landmarks)
                                                                             │
                                                                             ▼
                                                                 Feature Vector Normalization (73D)
                                                                             │
                                                                             ▼
                                                             Random Forest ML Classifier
                                                             (HELLO, HELP, YES, NO, STOP...)
                                                                             │
                                                                             ▼
                                                               Laptop Screen HUD + Speech Voice
```

---

## 🛠️ Step 1: Prepare Your Android Phone & Windows Laptop

### 1.1 Hardware Needed:
- Any Android Smartphone (Android 5.0 to Android 15+)
- USB Data Cable (use the original phone cable; ensure it supports **data transfer**, not charging only)
- Windows Laptop running Python 3.9 - 3.12

### 1.2 Enable USB Debugging on Android:
1. Open phone **Settings**.
2. Go to **About Phone** -> **Software Information**.
3. Tap **Build Number** **7 times** until you see *"You are now a developer!"*.
4. Go back to **Settings** -> **Developer Options**.
5. Enable the toggle for **USB Debugging**.
6. When plugging into the laptop, tap **"Always allow from this computer"** -> **Allow**.

---

## 🚀 Choose Your Connection Method

---

### METHOD A: Native Android 14+ USB Webcam (Easiest, No 3rd-Party Apps!)

*If your phone runs Android 14 or Android 15 (e.g. Google Pixel, Samsung Galaxy One UI 6+, Motorola, etc.):*

1. Connect phone to laptop with USB-C cable.
2. Swipe down the Android notification shade.
3. Tap the **"Charging this device via USB"** notification.
4. Under *"Use USB for"*, select **Webcam**.
5. Windows will play a device connection sound and immediately recognize your phone as a **Plug-and-Play USB Video Camera**!
6. It becomes **Camera Index 1** (or 2) in OpenCV without installing any software on phone or laptop!

---

### METHOD B: DroidCam USB via ADB (Universal, Works on ALL Android Versions)

*Recommended for all Android phones (ultra-reliable, zero latency, crystal clear HD):*

#### Step B1: Install Phone App
- Open Google Play Store and install: **DroidCam Webcam & OBS Camera** (Free by Dev47Apps).

#### Step B2: Install Laptop Client
- Download and run the **DroidCam Windows Client** from [dev47apps.com](https://www.dev47apps.com/).
- During install, let it install the virtual webcam drivers.

#### Step B3: Connect via USB
1. Open DroidCam on your phone.
2. Connect your phone via USB cable to laptop (ensure USB Debugging is ON).
3. Open DroidCam Client on Windows.
4. Click the **USB icon** in the DroidCam PC app.
5. Click the **Refresh** button (your phone model will appear in the dropdown).
6. Click **Start**.
7. Your phone camera feed is now broadcasting as a native Windows webcam (`DroidCam Source 2` or `3`)!

#### Pro Tip (No PC Client needed — Pure Python ADB Stream):
If you have `adb` installed, you don't even need the PC client:
```powershell
# Forward DroidCam's local phone port to your PC over USB cable:
adb forward tcp:4747 tcp:4747
```
Then in Python, OpenCV reads the USB-tunnel stream directly:
```python
cap = cv2.VideoCapture("http://127.0.0.1:4747/video")
```

---

### METHOD C: Iriun 4K Webcam via USB

1. Install **Iriun 4K Webcam** from Google Play Store on phone.
2. Install **Iriun Webcam for Windows** from [iriun.com](https://iriun.com/).
3. Connect phone via USB cable (USB Debugging enabled).
4. Launch Iriun on both devices. They automatically link over USB.
5. In OpenCV, Iriun is exposed as DirectShow Camera Index `1`.

---

## 🔍 How to Find Your Camera Index

Run this quick test in your terminal to see all detected cameras:

```powershell
python -c "import cv2; [print(f'Camera Index {i}: ACTIVE') for i in range(5) if cv2.VideoCapture(i, cv2.CAP_DSHOW).isOpened()]"
```

- **Index 0**: Default built-in laptop camera.
- **Index 1**: Your Android phone USB camera (or DroidCam / Iriun virtual camera).
- **Index 2**: External secondary camera (if multiple are attached).

---

## 💻 Running the AI Sign Language System with Phone Camera

### Option 1: Run Dedicated Phone Camera Recognition Script
We have created a dedicated script [`phone_cam_recognition.py`](../phone_cam_recognition.py) with automatic phone detection and live camera switching:

```powershell
# Automatically uses external phone camera (Index 1) if connected, or laptop camera
python phone_cam_recognition.py

# Explicitly specify camera index 1 (Phone USB)
python phone_cam_recognition.py --source 1

# If using DroidCam USB ADB stream
python phone_cam_recognition.py --source http://127.0.0.1:4747/video
```

**Live Hotkeys:**
- Press **`[C]`**: Instantly switch back and forth between Laptop Webcam and Phone Camera while running!
- Press **`[M]`**: Mute / Unmute voice output.
- Press **`[Q]`** or **`[ESC]`**: Exit cleanly.

---

### Option 2: Use Phone Camera in the Main Exhibition App
Open [`config.py`](../config.py) and change line 106:

```python
# Change from 0 (laptop) to 1 (phone):
CAMERA_INDEX = 1
```

Then run the main application:
```powershell
python main.py
```

---

### Option 3: Use Phone Camera in the Web Dashboard
With `CAMERA_INDEX = 1` in `config.py`, run:
```powershell
python dashboard_app.py
```
Open **`http://127.0.0.1:5000`** in your browser. The live stream will now be your smartphone camera!

---

## 🔧 Standalone Python Integration Code

Here is the complete, self-contained Python script to capture from phone USB, run MediaPipe hand tracking, and predict signs:

```python
import cv2
import joblib
import numpy as np
import collections

# Import project utilities
from utils.landmark_processor import LandmarkProcessor
from utils.hand_tracker import UniversalHandTracker
import config

def run_phone_recognition(camera_source=1):
    # 1. Load trained ML model & Label Encoder
    model = joblib.load(config.MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    print("Model loaded successfully.")

    # 2. Initialize MediaPipe Universal Hand Tracker
    hand_tracker = UniversalHandTracker(max_num_hands=1, min_detection_confidence=0.7)

    # 3. Open Phone Camera via USB
    print(f"Connecting to camera source {camera_source}...")
    if isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_source}. Trying laptop camera (0)...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    rolling_preds = collections.deque(maxlen=8)
    active_sign = "AWAITING GESTURE"

    print("Phone camera active! Hold hand in front of camera. Press [Q] to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w = frame.shape[:2]

        # 4. Detect 21 Hand Landmarks
        hands = hand_tracker.process(frame)
        current_pred = None
        conf = 0.0

        if hands:
            for hand in hands:
                # Draw skeletal lines & joints
                UniversalHandTracker.draw_landmarks(frame, hand)

                # 5. Extract 73D Invariant Feature Vector
                feat_vec = LandmarkProcessor.extract_feature_vector(hand)

                # 6. Random Forest Inference
                probs = model.predict_proba([feat_vec])[0]
                best_i = np.argmax(probs)
                conf = float(probs[best_i])
                label = label_encoder.classes_[best_i]

                if conf >= 0.70:
                    current_pred = label
                else:
                    current_pred = "Unknown Gesture"
                break

        # 7. Rolling Stability Smoothing
        if current_pred:
            rolling_preds.append(current_pred)
        elif len(rolling_preds) > 0:
            rolling_preds.popleft()

        if len(rolling_preds) == 8:
            common_sign, cnt = collections.Counter(rolling_preds).most_common(1)[0]
            if cnt >= 6:
                active_sign = common_sign
        elif not hands and len(rolling_preds) == 0:
            active_sign = "AWAITING GESTURE"

        # 8. Display Results on Screen
        color = (40, 40, 230) if active_sign in ["HELP", "STOP"] else (80, 200, 80)
        cv2.putText(frame, f"SIGN: {active_sign}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        cv2.putText(frame, f"CONFIDENCE: {int(conf * 100)}%", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Phone USB Camera - AI Sign Language", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()

if __name__ == "__main__":
    # Change to 0 for Laptop Webcam, 1 for Phone USB Webcam
    run_phone_recognition(camera_source=1)
```

---

## 🩺 Troubleshooting Guide

| Issue | Cause | Solution |
|---|---|---|
| **Camera Index 1 fails to open** | Phone app not streaming or unauthorized | Ensure phone app (DroidCam / Iriun) is launched and "Start" was pressed. Check that USB Debugging prompt was accepted. |
| **Black screen or frozen frame** | Windows privacy camera permissions | Open Windows Settings -> **Privacy & security** -> **Camera** -> Toggle ON *"Let desktop apps access your camera"*. |
| **Phone appears offline in DroidCam** | Missing USB OEM drivers | Download Google USB Driver or your phone manufacturer's USB driver (Samsung Smart Switch / Xiaomi / Pixel USB Driver). |
| **High latency or lag** | Wi-Fi mode selected instead of USB | Verify that the USB icon is selected in the client. A USB cable delivers stable 60 FPS with sub-5ms latency. |
| **MediaPipe does not detect hand** | Poor lighting or camera angle | Use a phone stand/tripod. Phone back cameras have superior autofocus and dynamic range compared to built-in laptop webcams. |
| **How to switch back to laptop webcam?** | Camera index set to 1 | Change `CAMERA_INDEX = 0` in `config.py` or press **`[C]`** in `phone_cam_recognition.py`. |
