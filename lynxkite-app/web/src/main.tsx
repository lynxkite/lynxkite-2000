import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./index.css";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Link,
  Route,
  RouterProvider,
  useRouteError,
} from "react-router";
import Code from "./Code.tsx";
import Directory from "./Directory.tsx";
import Workspace from "./workspace/Workspace.tsx";

function WorkspaceError() {
  const error = useRouteError();
  const stack = error instanceof Error ? error.stack : null;
  return (
    <div className="hero min-h-screen">
      <div className="card bg-base-100 shadow-sm">
        <div className="card-body">
          <h2 className="card-title">Something went wrong...</h2>
          <pre>{stack || "Unknown error."}</pre>
          <div className="card-actions justify-end">
            <Link to="/" className="btn btn-primary">
              Close workspace
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

const router = createBrowserRouter(
  createRoutesFromElements(
    <>
      <Route path="/" element={<Directory />} />
      <Route path="/dir" element={<Directory />} />
      <Route path="/dir/*" element={<Directory />} />
      <Route path="/edit/*" element={<Workspace />} errorElement={<WorkspaceError />} />
      <Route path="/code/*" element={<Code />} />
    </>,
  ),
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
