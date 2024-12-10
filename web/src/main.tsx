import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Directory from './Directory.tsx'
import Workspace from './workspace/Workspace.tsx'
import { BrowserRouter, Routes, Route } from "react-router";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Directory />} />
        <Route path="/dir" element={<Directory />} />
        <Route path="/dir/:path" element={<Directory />} />
        <Route path="/edit/:path" element={<Workspace />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
