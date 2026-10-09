# AI Sign Language Recognition (Web App)

## 1. Project Overview
This project is an advanced AI-powered web application that performs real-time Indian Sign Language (ISL) recognition. By combining modern web technologies (React/Vite/Three.js) with a localized Machine Learning backend (Python/Flask), it provides a robust, browser-native assistive communication tool.

**Note: All Arduino hardware integrations have been permanently removed. This is a pure web software solution.**

## 2. Features
- **Real-Time Recognition:** Classifies sign language instantly from your webcam.
- **42 Gesture Dictionary:** Robust support for 42 distinct gestures and phrases.
- **Web Speech API Voice Output:** Stable text-to-speech feedback via your browser.
- **3D Digital Exhibition Home Page:** A visually stunning interactive landing page built with Three.js.
- **Privacy-First:** All webcam tracking runs locally in Chrome; video streams are never uploaded.

## 3. Technology Stack
- **Frontend:** React, Vite, TailwindCSS, React-Three-Fiber (Three.js), React Router
- **Browser Computer Vision:** Google MediaPipe (Tasks Vision WASM)
- **Backend API:** Python 3, Flask, Flask-CORS
- **Machine Learning:** Scikit-Learn (Random Forest), Numpy

## 4. Installation
Ensure you have **Node.js (v18+)** and **Python (v3.10+)** installed on your system.

1. **Install Python dependencies:**
   ```bash
   pip install flask flask-cors opencv-python mediapipe numpy scikit-learn
   ```

2. **Install Frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

## 5. How to Run
The application runs as two connected services (Backend API + Frontend UI).

**Step 1: Start the Python Backend**
Open a terminal in the project root folder and run:
```bash
python app_web.py
```
*(Leave this terminal running. It will start the server on http://127.0.0.1:5001)*

**Step 2: Start the Web Frontend**
Open a SECOND terminal in the `frontend/` directory and run:
```bash
npm run dev
```

**Step 3: Open in Chrome**
Vite will provide a URL (usually `http://localhost:5173/`). Open this exact URL in **Google Chrome**.

## 6. How to Allow Camera Access
1. Navigate to the **RECOGNITION** page.
2. Click the green **START CAMERA** button.
3. Chrome will display a popup near the URL bar asking for camera permissions. Click **Allow**.
4. If you accidentally clicked Block, click the camera icon inside the right side of Chrome's URL bar, select "Always allow", and refresh the page.

## 7. How to Enable Voice
1. On the Recognition page dashboard, look for the **VOICE OUTPUT** card on the right.
2. Click the **VOICE: OFF** button to toggle it to **VOICE: ON**.
3. You can click **TEST VOICE** to ensure your browser's TTS engine is working.
4. Note: Voice only triggers when a gesture becomes stable. Holding a gesture will not spam repeated speech.

## 8. How the AI Model Works
1. **Webcam Capture:** Chrome accesses your camera via `navigator.mediaDevices.getUserMedia()`.
2. **MediaPipe Hand Landmarker (WASM):** Runs directly in the browser to extract 21 3D hand coordinates.
3. **Backend API:** The browser sends the (x,y,z) points to the Flask API.
4. **Feature Extraction:** Python normalizes the points (making them scale and translation invariant) and extracts 73 distinct geometric features.
5. **Random Forest:** The 73 features are fed into a pre-trained Scikit-Learn Random Forest model to predict one of 42 gesture classes.

## 9. Gesture Classes
The system recognizes 42 distinct phrases and signs. You can view the full interactive dictionary by clicking **EXPLORE GESTURES** on the Home page, or navigating to the **GESTURES** tab.

## 10. Troubleshooting
- **"AI model could not be reached"**: Ensure `python app_web.py` is currently running in a separate terminal, and there are no firewall blocks on port 5001.
- **"Camera unavailable"**: Ensure no other application (like Zoom or OBS) is using the webcam.
- **"Hand tracking could not be initialized"**: Ensure you have an active internet connection on the first run, as MediaPipe downloads WASM files from Google's CDN.

## 11. Browser Requirements
This application is strictly optimized for **Google Chrome (Desktop/Laptop)**.
Brave, Edge, or Firefox may work but are unsupported. Safari is explicitly incompatible due to differing Web Speech and WebRTC implementations.
