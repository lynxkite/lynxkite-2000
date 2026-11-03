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
  isRouteErrorResponse,
  Link,
  Route,
  RouterProvider,
  useRouteError,
} from "react-router";
import { isStaticWorkspaceMode, loadConfig } from "./common.ts";
import Directory from "./Directory.tsx";

const AuthCallback = lazy(() => import("./AuthCallback.tsx"));
const Code = lazy(() => import("./Code.tsx"));
const ProgressPage = lazy(() => import("./ProgressPage.tsx"));
const ProgressPageDemo = lazy(() => import("./ProgressPageDemo.tsx"));
const Workspace = lazy(() => import("./workspace/Workspace.tsx"));

function withSuspense(element: ReactElement) {
  return <Suspense>{element}</Suspense>;
}

function NotFound() {
  return (
    <div className="hero min-h-screen">
      <div className="card bg-base-100 shadow-sm">
        <div className="card-body">
          <h1 className="card-title">Page not found</h1>
          <p>The page you are looking for does not exist.</p>
          <div className="card-actions justify-end">
            <Link to="/" className="btn btn-primary">
              Go home
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function RouteError({ workspace = false }: { workspace?: boolean }) {
  const error = useRouteError();
  const details =
    error instanceof Error
      ? error.stack || error.message
      : isRouteErrorResponse(error)
        ? `${error.status} ${error.statusText}`
        : typeof error === "string"
          ? error
          : "Unknown error.";
  return (
    <div className="hero min-h-screen">
      <div className="card bg-base-100 shadow-sm">
        <div className="card-body">
          <h2 className="card-title">Something went wrong...</h2>
          <pre className="whitespace-pre-wrap break-words">{details}</pre>
          <div className="card-actions justify-end">
            <a href="/" className="btn btn-primary">
              {workspace ? "Close workspace" : "Go home"}
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

const router = createBrowserRouter(
  createRoutesFromElements(
    isStaticWorkspaceMode() ? (
      <Route
        path="*"
        element={withSuspense(<Workspace />)}
        errorElement={<RouteError workspace />}
      />
    ) : (
      <Route errorElement={<RouteError />}>
        <Route path="/" element={<Directory />} />
        <Route path="/dir" element={<Directory />} />
        <Route path="/dir/*" element={<Directory />} />
        <Route path="/auth/callback" element={withSuspense(<AuthCallback />)} />
        <Route
          path="/edit/*"
          element={withSuspense(<Workspace />)}
          errorElement={<RouteError workspace />}
        />
        <Route path="/code/*" element={withSuspense(<Code />)} />
        <Route path="/progress" element={withSuspense(<ProgressPage />)} />
        <Route path="/progress-demo" element={withSuspense(<ProgressPageDemo />)} />
        <Route path="*" element={<NotFound />} />
      </Route>
    ),
  ),
);

const root = createRoot(document.getElementById("root")!);

async function bootstrap() {
  try {
    await loadConfig();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to reach the server.";
    root.render(
      <div className="hero min-h-screen">
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body">
            <h2 className="card-title">LynxKite failed to load</h2>
            <p>{message}</p>
          </div>
        </div>
      </div>,
    );
    return;
  }
  root.render(
    <StrictMode>
      <RouterProvider router={router} />
      <ReactTooltip id="tooltip-global" opacity={1} />
    </StrictMode>,
  );
}

void bootstrap();
