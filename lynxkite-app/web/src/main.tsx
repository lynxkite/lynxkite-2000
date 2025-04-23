import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./index.css";
import { BrowserRouter, Route, Routes } from "react-router";
import Code from "./Code.tsx";
import Directory from "./Directory.tsx";
import Workspace from "./workspace/Workspace.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Directory />} />
        <Route path="/dir" element={<Directory />} />
        <Route path="/dir/*" element={<Directory />} />
        <Route path="/edit/*" element={<Workspace />} />
        <Route path="/code/*" element={<Code />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
