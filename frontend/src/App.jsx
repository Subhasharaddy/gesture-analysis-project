import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Recognition from './pages/Recognition';
import Gestures from './pages/Gestures';
import About from './pages/About';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-950 text-white font-sans flex flex-col">
        <nav className="bg-gray-900 border-b border-cyan-900/50 p-4 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <Link to="/" className="text-xl font-bold text-cyan-400 tracking-wider">
              AI SIGN LANGUAGE
            </Link>
            <div className="flex gap-6">
              <Link to="/" className="hover:text-cyan-400 transition">HOME</Link>
              <Link to="/recognition" className="hover:text-cyan-400 transition">RECOGNITION</Link>
              <Link to="/gestures" className="hover:text-cyan-400 transition">GESTURES</Link>
              <Link to="/about" className="hover:text-cyan-400 transition">ABOUT</Link>
            </div>
          </div>
        </nav>
        <main className="flex-1 flex flex-col relative overflow-hidden">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/recognition" element={<Recognition />} />
            <Route path="/gestures" element={<Gestures />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
