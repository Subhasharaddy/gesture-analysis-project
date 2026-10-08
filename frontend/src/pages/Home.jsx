import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Sphere } from '@react-three/drei';
import { useNavigate } from 'react-router-dom';

const base_nodes = [
  [0, 1.5, 0],       // 0: Wrist
  [-0.4, 0.5, 0.2],  // 1: Thumb base
  [-0.6, 0.2, 0.3],  // 2: Thumb mid
  [-0.8, -0.2, 0.4], // 3: Thumb tip
  [-0.3, 0, 0],      // 4: Index base
  [-0.4, -0.6, 0],   // 5: Index mid
  [-0.5, -1.2, 0],   // 6: Index tip
  [0, -0.1, 0],      // 7: Middle base
  [0, -0.8, 0],      // 8: Middle mid
  [0, -1.4, 0],      // 9: Middle tip
  [0.3, 0, 0],       // 10: Ring base
  [0.4, -0.7, 0],    // 11: Ring mid
  [0.5, -1.3, 0],    // 12: Ring tip
  [0.5, 0.3, -0.1],  // 13: Pinky base
  [0.6, -0.3, -0.2], // 14: Pinky mid
  [0.7, -0.8, -0.3], // 15: Pinky tip
];

const edges = [
  [0,1], [1,2], [2,3],
  [0,4], [4,5], [5,6],
  [0,7], [7,8], [8,9],
  [0,10], [10,11], [11,12],
  [0,13], [13,14], [14,15],
  [4,7], [7,10], [10,13] // Palm connections
];

function HolographicHand() {
  const groupRef = useRef();
  
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    groupRef.current.rotation.y = t * 0.2;
    groupRef.current.position.y = Math.sin(t * 1.5) * 0.1;
  });

  return (
    <group ref={groupRef} scale={[1.2, 1.2, 1.2]}>
      {edges.map((edge, i) => (
        <Line
          key={`edge-${i}`}
          points={[base_nodes[edge[0]], base_nodes[edge[1]]]}
          color="#22d3ee"
          lineWidth={1.5}
          transparent
          opacity={0.6}
        />
      ))}
      
      {base_nodes.map((node, i) => (
        <Sphere key={`node-${i}`} args={[0.04, 16, 16]} position={node}>
          <meshBasicMaterial color="#ffffff" />
          <pointLight color="#22d3ee" distance={0.5} intensity={0.5} />
        </Sphere>
      ))}
    </group>
  );
}

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="flex-1 flex flex-col md:flex-row items-center justify-center p-8 lg:p-24 relative">
      {/* Background elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-900/10 to-purple-900/10 pointer-events-none" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />

      {/* Left side text */}
      <div className="flex-1 z-10 max-w-2xl">
        <h2 className="text-cyan-400 font-bold tracking-[0.2em] mb-4 text-sm md:text-base">AI EXHIBITION PROJECT</h2>
        <h1 className="text-5xl md:text-7xl font-black mb-6 glow-text leading-tight">
          AI SIGN LANGUAGE <br /> RECOGNITION
        </h1>
        <p className="text-xl md:text-2xl text-gray-400 mb-10 italic border-l-4 border-cyan-500 pl-4">
          Breaking Communication Barriers with Artificial Intelligence
        </p>
        
        <ul className="mb-12 space-y-4">
          <li className="flex items-center text-lg font-semibold text-emerald-400">
            <span className="mr-3 text-2xl">✦</span> 42 Gesture Classes
          </li>
          <li className="flex items-center text-lg font-semibold text-cyan-400">
            <span className="mr-3 text-2xl">✦</span> Real-Time AI Inference
          </li>
          <li className="flex items-center text-lg font-semibold text-purple-400">
            <span className="mr-3 text-2xl">✦</span> Browser-Native Voice Output
          </li>
        </ul>

        <div className="flex gap-4 flex-wrap">
          <button 
            onClick={() => navigate('/recognition')}
            className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white px-8 py-4 rounded-full font-bold shadow-[0_0_20px_rgba(34,211,238,0.4)] transition-transform hover:scale-105 uppercase tracking-wider"
          >
            Start Recognition
          </button>
          <button 
            onClick={() => navigate('/gestures')}
            className="glass-card text-white px-8 py-4 rounded-full font-bold hover:bg-gray-800 transition-colors uppercase tracking-wider"
          >
            Explore Gestures
          </button>
        </div>
      </div>

      {/* Right side 3D Canvas */}
      <div className="flex-1 h-[50vh] md:h-[80vh] w-full z-10 relative mt-12 md:mt-0">
        <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
          <ambientLight intensity={0.2} />
          <pointLight position={[10, 10, 10]} intensity={1} color="#22d3ee" />
          <Stars radius={100} depth={50} count={2000} factor={4} saturation={0} fade speed={1} />
          <HolographicHand />
          <OrbitControls enableZoom={false} enablePan={false} autoRotate={false} />
        </Canvas>
      </div>
    </div>
  );
}
