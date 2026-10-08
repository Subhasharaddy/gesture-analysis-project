import React, { useEffect, useState } from 'react';

export default function Gestures() {
  const [gestures, setGestures] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/gestures")
      .then(res => res.json())
      .then(data => {
        setGestures(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError("Failed to load gesture guide. Ensure Python backend is running.");
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-12 text-center text-cyan-400 font-bold">Loading gesture dictionary...</div>;
  if (error) return <div className="p-12 text-center text-red-400 font-bold">{error}</div>;

  return (
    <div className="p-8 max-w-7xl mx-auto w-full h-full overflow-y-auto">
      <div className="text-center mb-12">
        <h2 className="text-cyan-400 font-bold tracking-[0.2em] mb-2 text-sm uppercase">Comprehensive Reference</h2>
        <h1 className="text-4xl md:text-5xl font-black mb-4 text-white">42 GESTURE CLASSES</h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          This system uses a Random Forest ML model trained on 42 unique Indian Sign Language (ISL) gestures. 
          Below is the complete dictionary of recognized signs.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-20">
        {Object.entries(gestures).map(([name, info], idx) => (
          <div key={name} className="glass-card rounded-xl p-6 transition-all hover:-translate-y-1">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-2xl font-black text-white">{idx + 1}. {name}</h3>
              {info.is_emergency && (
                <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded text-xs font-bold border border-red-500/50">
                  EMERGENCY
                </span>
              )}
            </div>
            
            <div className="space-y-2">
              <p className="text-sm">
                <span className="text-gray-500 font-bold block mb-1">MEANING:</span>
                <span className="text-cyan-400">{info.meaning || info.english || name}</span>
              </p>
              
              {info.kannada && (
                <p className="text-sm">
                  <span className="text-gray-500 font-bold block mb-1">KANNADA:</span>
                  <span className="text-emerald-400 font-bold text-lg">{info.kannada}</span> 
                  <span className="text-gray-400 ml-2">({info.kannada_translit})</span>
                </p>
              )}
              
              <p className="text-sm mt-4 pt-4 border-t border-gray-800">
                <span className="text-gray-500 font-bold block mb-1">HOW TO PERFORM:</span>
                <span className="text-gray-300">{info.how_to_perform || info.description || "N/A"}</span>
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
