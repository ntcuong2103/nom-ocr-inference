import { Link, Route, Routes } from 'react-router-dom'
import { PageNavSidebar } from './components/Sidebar/PageNavSidebar'
import { PageAnnotationView } from './components/PageView/PageAnnotationView'
import { GalleryView } from './components/Gallery/GalleryView'
import './App.css'

function Welcome() {
  return (
    <div className="welcome">
      <p>Select a page from the sidebar, or browse the character gallery.</p>
      <Link to="/gallery">Open Gallery</Link>
    </div>
  )
}

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar-pane">
        <div className="brand">
          <span className="nom">NomnaOCR</span> Annotator
        </div>
        <nav className="top-nav">
          <Link to="/gallery">Gallery</Link>
        </nav>
        <PageNavSidebar />
      </aside>
      <main className="main-pane">
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/page/:volume/:page" element={<PageAnnotationView />} />
          <Route path="/gallery" element={<GalleryView />} />
        </Routes>
      </main>
    </div>
  )
}
