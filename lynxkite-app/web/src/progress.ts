import { useEffect, useState } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import { apiFetch } from "./common.ts";

export function formatWorkspaceEta(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds <= 0) return "done";
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  if (minutes > 0) return `~${minutes}m ${remainingSeconds}s left`;
  return `~${remainingSeconds}s left`;
}

export function workspaceProgressColor(status: string): "primary" | "error" | "neutral" {
  if (status === "active" || status === "running") return "primary";
  if (status === "failed") return "error";
  return "neutral";
}

export function formatWorkspaceProgressSuffix(workspace: {
  status?: string;
  etaSeconds?: number | null;
  boxesFailed?: number;
}): string {
  if (workspace.status === "done") return "done";
  if (workspace.status === "failed") return `${workspace.boxesFailed ?? 0} failed`;
  return formatWorkspaceEta(workspace.etaSeconds);
}

export function parseProgressWorkspace(value: unknown): any | null {
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (value && typeof value === "object") {
    return value as any;
  }
  return null;
}

/** Normalize backend progress payload for UI components. */
export function getWorkspaceProgress(workspace: any) {
  const boxesDone = Number(workspace?.boxes_done ?? 0);
  const boxesFailed = Number(workspace?.boxes_failed ?? 0);
  const boxesTotal = Number(workspace?.boxes_total ?? 0);
  const fraction = Math.max(0, Math.min(1, Number(workspace?.progress_fraction ?? 0)));

  return {
    activeNode: workspace?.active_node ?? null,
    boxesDone,
    boxesFailed,
    boxesTotal,
    elapsedSeconds:
      typeof workspace?.elapsed_seconds === "number" ? workspace.elapsed_seconds : null,
    fraction,
    paused: Boolean(workspace?.paused),
    percent: fraction * 100,
    status: workspace?.status ?? "",
    etaSeconds: typeof workspace?.eta_seconds === "number" ? workspace.eta_seconds : null,
  };
}

async function postProgressAction(path: string, body: object, what: string) {
  try {
    const res = await apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) console.error(`Failed to ${what}: ${res.status} ${res.statusText}`);
  } catch (e) {
    console.error(`Failed to ${what}`, e);
  }
}

export function pauseWorkspace(roomName: string, paused: boolean) {
  return postProgressAction(
    "/api/pause_workspace",
    { room_name: roomName, paused },
    "pause workspace",
  );
}

export function stopWorkspace(roomName: string) {
  return postProgressAction("/api/stop_workspace", { room_name: roomName }, "stop workspace");
}

/**
 * Subscribes to the deployment-wide progress CRDT. Pass `enabled: false` to skip connecting.
 * Also re-renders every second so the time-left estimates count down.
 */
export function useProgress(enabled: boolean) {
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [gpuServices, setGpuServices] = useState<any[]>([]);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    const doc = new Y.Doc();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const provider = new WebsocketProvider(
      `${proto}//${location.host}/ws/progress/crdt`,
      "progress",
      doc,
    );
    const wsMap = doc.getMap("workspaces");
    const gpuServicesText = doc.getText("gpu_services");

    function syncWorkspaces() {
      const ws: any[] = [];
      for (const value of (wsMap as Y.Map<unknown>).values()) {
        const parsed = parseProgressWorkspace(value);
        if (parsed) ws.push(parsed);
      }
      setWorkspaces(ws);
    }

    function syncGpuServices() {
      const raw = gpuServicesText.toString();
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setGpuServices(parsed);
      } catch {
        // Ignore malformed JSON during partial updates.
      }
    }

    function syncAll() {
      syncWorkspaces();
      syncGpuServices();
    }

    wsMap.observe(syncWorkspaces);
    gpuServicesText.observe(syncGpuServices);
    provider.on("sync", syncAll);
    provider.on("status", (event: any) => {
      if (event?.status === "connected") syncAll();
    });

    return () => {
      wsMap.unobserve(syncWorkspaces);
      gpuServicesText.unobserve(syncGpuServices);
      provider.destroy();
      doc.destroy();
    };
  }, [enabled]);

  return { workspaces, gpuServices };
}
