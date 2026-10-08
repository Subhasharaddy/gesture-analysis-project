import React from 'react';

export default function About() {
  return (
    <div className="p-8 max-w-4xl mx-auto w-full h-full overflow-y-auto">
      <div className="glass-card rounded-2xl p-10 mb-20">
        <h1 className="text-4xl md:text-5xl font-black mb-8 glow-text">ABOUT THE PROJECT</h1>
        
        <div className="space-y-8 text-gray-300 text-lg leading-relaxed">
          <section>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4 border-b border-gray-800 pb-2">1. Architecture Overview</h2>
            <p>
              This is a state-of-the-art edge AI web application designed to bridge the communication gap.
              It operates entirely within Google Chrome, leveraging modern web technologies alongside a localized
              Python Machine Learning backend.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4 border-b border-gray-800 pb-2">2. Computer Vision & MediaPipe</h2>
            <p>
              The application accesses the user's local webcam via the browser's native <code className="text-emerald-400">navigator.mediaDevices.getUserMedia()</code> API.
              Video frames are passed into <strong>MediaPipe Hand Landmarker</strong>, which uses Google's highly optimized WebAssembly (WASM) implementation to extract <strong>21 3D hand landmarks</strong> in real-time.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4 border-b border-gray-800 pb-2">3. Machine Learning Inference</h2>
            <p>
              The 21 extracted (x,y,z) spatial coordinates are sent to the local Python Flask backend. 
              The backend normalizes these coordinates to ensure scale and translation invariance, then extracts 73 distinct geometric features (like finger spans and joint distances).
              These features are evaluated against a pre-trained <strong>Random Forest Classifier</strong> mapping to exactly 42 distinct gesture classes.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4 border-b border-gray-800 pb-2">4. Web Speech API (Voice Output)</h2>
            <p>
              Once a gesture is stabilized using majority-voting history, it triggers the browser's native <code className="text-emerald-400">window.speechSynthesis</code> API.
              This provides immediate, clear auditory feedback. The application implements smart debouncing so you don't hear a word repeated endlessly while holding a gesture.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4 border-b border-gray-800 pb-2">5. Privacy & Security</h2>
            <p>
              All processing happens <strong>locally</strong>. Video frames never leave your browser, and the Python backend runs strictly on localhost. 
              The application requires explicit camera permission to operate, and no audio or video data is ever stored.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
