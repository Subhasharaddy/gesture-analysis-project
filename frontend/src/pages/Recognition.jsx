import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

export default function Recognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const cameraActiveRef = useRef(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [status, setStatus] = useState("INITIALIZING");
  const [currentGesture, setCurrentGesture] = useState("");
  const [animateEmoji, setAnimateEmoji] = useState(false);
  
  const GESTURE_EMOJIS = {
    "OPEN PALM": "🖐️", "CLOSED FIST": "✊", "INDEX FINGER POINT": "☝️", "PEACE SIGN": "✌️",
    "THUMBS UP": "👍", "BAD": "👎", "BYE": "👋", "CALL FOR HELP": "🆘", "COME": "🫴",
    "DOCTOR": "🩺", "EMERGENCY": "🚨", "FIST": "✊", "FOOD": "🍎", "FRIEND": "🤝",
    "GO": "👉", "GOOD": "👍", "GOOD NIGHT": "🌙", "HELLO": "👋", "HELP": "🙋",
    "I / ME": "👈", "LOVE": "🤟", "I AM FINE": "👌", "NAMASTE": "🙏", "PHONE": "🤙",
    "STOP": "✋", "WAIT": "✋", "YES": "✊", "NO": "🙅", "THANK YOU": "🙏",
    "PLEASE": "🤲", "SORRY": "🥺", "WELCOME": "🤗", "WATER": "💧", "EAT": "🍽️",
    "SLEEP": "😴", "HOME": "🏠", "COME HERE": "👋", "DANGER": "⚠️", "VICTORY": "✌️",
    "ROCK ON": "🤘", "POINT": "☝️", "PEACE": "✌️"
  };
  
  const [prediction, setPrediction] = useState({
    gesture: "No hand detected",
    gesture_name: "No hand detected",
    confidence: 0,
    kannada: "",
    meaning: ""
  });
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [voiceStatus, setVoiceStatus] = useState("READY");
  
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  
  const handLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const requestRef = useRef(null);
  const streamRef = useRef(null);
  const isPredictingRef = useRef(false);
  const lastSpokenSignRef = useRef(null);
  
  useEffect(() => {
    if (prediction.gesture_name && prediction.gesture_name !== currentGesture) {
      setCurrentGesture(prediction.gesture_name);
      setAnimateEmoji(true);
      const timer = setTimeout(() => setAnimateEmoji(false), 300);
      return () => clearTimeout(timer);
    }
  }, [prediction.gesture_name, currentGesture]);

  const fetchCameras = async () => {
    try {
      const devicesInfo = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devicesInfo.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0) {
        // If we haven't selected one yet, select the first one
        setSelectedDeviceId(prev => prev || videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error("Error fetching cameras:", err);
    }
  };
  
  useEffect(() => {
    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        setStatus("READY TO START");
      } catch (err) {
        console.error("MediaPipe Init Error:", err);
        setErrorMsg("Hand tracking could not be initialized.");
      }
    }
    initMediaPipe();
    
    return () => {
      stopCamera();
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  useEffect(() => {
    fetchCameras();
  }, []);

  const startCamera = async () => {
    setErrorMsg(null);
    stopCamera(); // Stop previous stream before starting a new one
    try {
      const constraints = selectedDeviceId 
        ? { video: { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } }
        : { video: { width: 640, height: 480, facingMode: "user" } };
        
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        cameraActiveRef.current = true;
        setCameraActive(true);
        setStatus("RECOGNIZING");
        predictWebcam();
      };
      
      // Fetch cameras again now that permission is granted (to get labels)
      fetchCameras();
    } catch (err) {
      console.error("Camera Error:", err);
      setErrorMsg("Camera permission required, or camera is in use by another app.");
    }
  };

  const stopCamera = () => {
    cameraActiveRef.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setStatus("STOPPED");
    cancelAnimationFrame(requestRef.current);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const speakGesture = (text) => {
    if (!voiceEnabled || !window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel(); // Cancel any ongoing stale speech
      setVoiceStatus("SPEAKING...");
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = () => setVoiceStatus("READY");
      utterance.onerror = () => setVoiceStatus("READY");
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.error("Speech synthesis error:", err);
      setVoiceStatus("ERROR");
    }
  };

  const testVoice = () => {
    if (!window.speechSynthesis) {
      setVoiceStatus("ERROR");
      return;
    }
    setVoiceStatus("SPEAKING...");
    const utterance = new SpeechSynthesisUtterance("AI sign language recognition voice test successful.");
    utterance.onend = () => setVoiceStatus("READY");
    utterance.onerror = () => setVoiceStatus("ERROR");
    window.speechSynthesis.speak(utterance);
  };

  const predictWebcam = async () => {
    const video = videoRef.current;
    if (!video || !cameraActiveRef.current || !handLandmarkerRef.current) return;

    let startTimeMs = performance.now();
    if (lastVideoTimeRef.current !== video.currentTime) {
      lastVideoTimeRef.current = video.currentTime;
      
      const results = handLandmarkerRef.current.detectForVideo(video, startTimeMs);
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const hasHands = results.landmarks && results.landmarks.length > 0;
      if (hasHands) {
        // Draw landmarks
        for (const landmarks of results.landmarks) {
          drawConnectors(ctx, landmarks, [
            [0,1], [1,2], [2,3], [3,4],
            [0,5], [5,6], [6,7], [7,8],
            [5,9], [9,10], [10,11], [11,12],
            [9,13], [13,14], [14,15], [15,16],
            [13,17], [0,17], [17,18], [18,19], [19,20]
          ], {color: '#22d3ee', lineWidth: 2});
          drawLandmarks(ctx, landmarks, {color: '#ffffff', lineWidth: 1, radius: 3});
        }
      }

      // Send to Backend (with in-flight guard to eliminate request race conditions and UI jitter)
      if (!isPredictingRef.current) {
        isPredictingRef.current = true;
        try {
          const handsData = hasHands ? results.landmarks.map((lms, idx) => ({
            label: results.handednesses?.[idx]?.[0]?.categoryName || "Right",
            landmarks: lms.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }))
          })) : [];

          const response = await fetch("http://localhost:5001/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              hands: handsData,
              image_shape: [video.videoHeight, video.videoWidth]
            })
          });
          const data = await response.json();
          const stableName = data.gesture_name || data.gesture || "No hand detected";

          if (stableName && stableName !== "No hand detected" && stableName !== "Detecting..." && stableName !== "Unknown Gesture") {
            setPrediction({
              ...data,
              gesture: stableName,
              gesture_name: stableName
            });

            // Trigger speech once per stable gesture switch
            if (stableName !== lastSpokenSignRef.current) {
              lastSpokenSignRef.current = stableName;
              speakGesture(data.spoken_phrase || stableName);
            }
          } else {
            // Handle "No hand detected", "Detecting...", "Unknown Gesture"
            if (stableName === "No hand detected") {
              lastSpokenSignRef.current = null;
            }
            setPrediction(p => ({
              ...p,
              ...data,
              gesture: stableName,
              gesture_name: stableName,
              confidence: data.confidence || 0
            }));
          }
        } catch (err) {
          console.error("Backend API Error:", err);
          setErrorMsg("AI model could not be reached (Is Python backend running?).");
        } finally {
          isPredictingRef.current = false;
        }
      }
    }
    
    if (cameraActiveRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  };

  const drawConnectors = (ctx, landmarks, edges, style) => {
    ctx.save();
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.lineWidth;
    for (const edge of edges) {
      const p1 = landmarks[edge[0]];
      const p2 = landmarks[edge[1]];
      ctx.beginPath();
      ctx.moveTo(p1.x * ctx.canvas.width, p1.y * ctx.canvas.height);
      ctx.lineTo(p2.x * ctx.canvas.width, p2.y * ctx.canvas.height);
      ctx.stroke();
    }
    ctx.restore();
  };
  
  const drawLandmarks = (ctx, landmarks, style) => {
    ctx.save();
    ctx.fillStyle = style.color;
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * ctx.canvas.width, lm.y * ctx.canvas.height, style.radius, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  };

  return (
    <div className="p-6 h-full flex flex-col md:flex-row gap-6 relative z-10">
      
      {/* Left side: Camera View */}
      <div className="flex-1 glass-card rounded-2xl p-4 flex flex-col relative overflow-hidden">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse mr-2"></span>
          LIVE CAMERA
        </h2>
        
        {errorMsg && (
          <div className="bg-red-500/20 border border-red-500 text-red-200 p-4 rounded-lg mb-4">
            {errorMsg}
          </div>
        )}
        
        <div className="relative w-full aspect-video bg-gray-900 rounded-lg overflow-hidden border border-gray-700 flex items-center justify-center shadow-inner">
          {!cameraActive && !errorMsg && (
            <span className="text-gray-500 uppercase tracking-widest text-sm">Camera Offline</span>
          )}
          
          <video 
            ref={videoRef} 
            className="absolute inset-0 w-full h-full object-cover" 
            style={{ transform: "scaleX(-1)", display: cameraActive ? "block" : "none" }}
            playsInline 
            autoPlay 
            muted 
          />
          <canvas 
            ref={canvasRef} 
            className="absolute inset-0 w-full h-full object-cover pointer-events-none" 
            width={640} height={480}
            style={{ transform: "scaleX(-1)", display: cameraActive ? "block" : "none" }}
          />
        </div>
        
        <div className="mt-6 flex flex-wrap gap-4 items-center">
          <select 
            className="bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2 outline-none focus:border-cyan-500 transition max-w-[250px] truncate"
            value={selectedDeviceId}
            onChange={(e) => {
              setSelectedDeviceId(e.target.value);
              if (cameraActive) {
                 // The useEffect doesn't automatically restart, we need to let user hit start or we can auto-restart.
                 // It's cleaner to just let the user hit START again or auto-restart. We'll handle it via state/effect or manual.
              }
            }}
          >
            {devices.length === 0 && <option value="">No cameras found</option>}
            {devices.map((device, idx) => (
              <option key={device.deviceId || idx} value={device.deviceId}>
                {device.label || `Camera ${idx + 1}`}
              </option>
            ))}
          </select>

          <button 
            onClick={startCamera} 
            className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-full font-semibold transition"
          >
            {cameraActive ? "SWITCH CAMERA" : "START CAMERA"}
          </button>
          
          <button 
            onClick={stopCamera} 
            disabled={!cameraActive}
            className="bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white px-6 py-2 rounded-full font-semibold transition"
          >
            STOP CAMERA
          </button>
        </div>
      </div>
      
      {/* Right side: Dashboard */}
      <div className="w-full md:w-[400px] flex flex-col gap-6">
        
        {/* Main Recognition Card */}
        <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-emerald-500"></div>
          <h2 className="text-gray-400 text-sm font-bold tracking-wider mb-2 uppercase">AI Recognition</h2>
          
          <div className="my-6 text-center">
            <div className="text-sm text-gray-500 mb-1">GESTURE</div>
            <div className={`text-4xl font-black transition-opacity duration-300 ${prediction.gesture_name && prediction.gesture_name !== "No hand detected" && prediction.gesture_name !== "Hold gesture steady" ? "text-white glow-text" : "text-gray-600"}`}>
              {prediction.gesture_name || prediction.gesture || "No hand detected"}
            </div>
            
            {prediction.kannada && (
              <div className="text-xl text-emerald-400 mt-2 font-bold">{prediction.kannada}</div>
            )}
            
            {/* Gesture Emoji Visualization */}
            {prediction.gesture_name && prediction.gesture_name !== "No hand detected" && prediction.gesture_name !== "Hold gesture steady" && prediction.gesture_name !== "Detecting..." && prediction.gesture_name !== "Unknown Gesture" && (
              <div className={`text-7xl mt-6 transition-transform duration-300 ease-out ${animateEmoji ? 'scale-125' : 'scale-100'}`}>
                {GESTURE_EMOJIS[prediction.gesture_name] || "✨"}
              </div>
            )}
          </div>
          
          <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-400 font-bold">CONFIDENCE</span>
              <span className="text-lg font-bold text-cyan-400">{Math.round(prediction.confidence * 100)}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all duration-300"
                style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
              ></div>
            </div>
          </div>
          
          <div className="mt-4 flex justify-between items-center">
            <span className="text-sm text-gray-400 font-bold">STATUS</span>
            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
              status === "RECOGNIZING" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" : "bg-gray-800 text-gray-400"
            }`}>
              {status}
            </span>
          </div>
        </div>
        
        {/* Voice Control Card */}
        <div className="glass-card rounded-2xl p-6">
          <h2 className="text-gray-400 text-sm font-bold tracking-wider mb-4 uppercase flex justify-between items-center">
            Voice Output
            <span className={`text-xs ${voiceStatus === "ERROR" ? "text-red-400" : (voiceStatus === "SPEAKING..." ? "text-cyan-400 animate-pulse" : "text-gray-500")}`}>
              {voiceStatus}
            </span>
          </h2>
          
          <div className="flex flex-col gap-3">
            <button 
              onClick={() => setVoiceEnabled(!voiceEnabled)}
              className={`py-3 rounded-xl font-bold transition flex justify-center items-center gap-2 ${
                voiceEnabled ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/50" : "bg-gray-800 text-gray-400 border border-gray-700 hover:bg-gray-700"
              }`}
            >
              🔊 VOICE: {voiceEnabled ? "ON" : "OFF"}
            </button>
            <button 
              onClick={testVoice}
              className="py-2 bg-purple-500/20 text-purple-400 border border-purple-500/50 hover:bg-purple-500/30 rounded-xl font-bold transition"
            >
              TEST VOICE
            </button>
          </div>
        </div>
        
      </div>
    </div>
  );
}
