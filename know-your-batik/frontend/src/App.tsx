import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar.tsx'
import Footer from './components/Footer.tsx'
import Home from './pages/Home.tsx'
import Classifier from './pages/Classifier.tsx'
import Gallery from './pages/Gallery.tsx'
import Learn from './pages/Learn.tsx'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/classifier" element={<Classifier />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/learn" element={<Learn />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}
