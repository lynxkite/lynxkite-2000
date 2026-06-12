import { lazy, type ReactElement, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { Tooltip as ReactTooltip } from "react-tooltip";
import "@fontsource/inter";
import "@fontsource/inter/500.css";
import "@fontsource/inter/700.css";
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
import { useConfig } from "./common.ts";
import Directory from "./Directory.tsx";

function ConfigGuard({ children }: { children: ReactElement }) {
  const config = useConfig();
  if (!config.data) return null;
  return children;
}

const AuthCallback = lazy(() => import("./AuthCallback.tsx"));
const Code = lazy(() => import("./Code.tsx"));
const ProgressPage = lazy(() => import("./ProgressPage.tsx"));
const ProgressPageDemo = lazy(() => import("./ProgressPageDemo.tsx"));
const Workspace = lazy(() => import("./workspace/Workspace.tsx"));

function withSuspense(element: ReactElement) {
  return <Suspense>{element}</Suspense>;
}

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
      <Route
        path="/"
        element={
          <ConfigGuard>
            <Directory />
          </ConfigGuard>
        }
      />
      <Route
        path="/dir"
        element={
          <ConfigGuard>
            <Directory />
          </ConfigGuard>
        }
      />
      <Route
        path="/dir/*"
        element={
          <ConfigGuard>
            <Directory />
          </ConfigGuard>
        }
      />
      <Route path="/auth/callback" element={withSuspense(<AuthCallback />)} />
      <Route
        path="/edit/*"
        element={<ConfigGuard>{withSuspense(<Workspace />)}</ConfigGuard>}
        errorElement={<WorkspaceError />}
      />
      <Route path="/code/*" element={<ConfigGuard>{withSuspense(<Code />)}</ConfigGuard>} />
      <Route
        path="/progress"
        element={<ConfigGuard>{withSuspense(<ProgressPage />)}</ConfigGuard>}
      />
      <Route
        path="/progress-demo"
        element={<ConfigGuard>{withSuspense(<ProgressPageDemo />)}</ConfigGuard>}
      />
    </>,
  ),
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
    <ReactTooltip id="tooltip-global" opacity={1} />
  </StrictMode>,
);
