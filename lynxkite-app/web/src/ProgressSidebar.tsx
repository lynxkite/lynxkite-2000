// Compact workspace progress list embedded in the directory sidebar.
import { useEffect, useState } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import Pause from "~icons/tabler/player-pause-filled";
import Play from "~icons/tabler/player-play-filled";
import Stop from "~icons/tabler/player-stop-filled";

function timeLeft(ws: any): string {
  if (ws.eta_seconds == null) return "";
  if (ws.eta_seconds <= 0) return "done";
  const minutes = Math.floor(ws.eta_seconds / 60);
  const seconds = Math.floor(ws.eta_seconds % 60);
  if (minutes > 0) return `~${minutes}m ${seconds}s left`;
  return `~${seconds}s left`;
}

export function WorkspaceProgressList() {
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [, setTick] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const doc = new Y.Doc();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const provider = new WebsocketProvider(
      `${proto}//${location.host}/ws/progress/crdt`,
      "progress",
      doc,
    );
    const wsMap = doc.getMap("workspaces");

    function sync() {
      const ws: any[] = [];
      for (const value of (wsMap as Y.Map<unknown>).values()) {
        try {
          ws.push(typeof value === "string" ? JSON.parse(value) : (value as any));
        } catch {}
      }
      setWorkspaces(ws);
    }

    wsMap.observe(sync);
    provider.on("sync", sync);
    provider.on("status", (event: any) => {
      if (event?.status === "connected") sync();
    });

    return () => {
      wsMap.unobserve(sync);
      provider.destroy();
      doc.destroy();
    };
  }, []);

  async function pauseWorkspace(roomName: string, paused: boolean) {
    await fetch("/api/pause_workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room_name: roomName, paused }),
    });
  }

  async function stopWorkspace(roomName: string) {
    await fetch("/api/stop_workspace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room_name: roomName }),
    });
  }

  return (
    <div className="sidebar-progress-section">
      <div className="sidebar-section-title">Running workspaces</div>
      {workspaces.length === 0 ? (
        <p className="sidebar-empty">No workspaces running.</p>
      ) : (
        workspaces.map((ws) => {
          const roomName = ws.room_name || ws.name;
          const boxFraction = ws.boxes_total > 0 ? ws.boxes_done / ws.boxes_total : 0;
          const tqdm = ws.active_node?.tqdm;
          const tqdmFraction = tqdm?.total > 0 ? tqdm.n / tqdm.total : null;
          const progress =
            tqdmFraction != null ? (ws.boxes_done + tqdmFraction) / ws.boxes_total : boxFraction;
          return (
            <div key={ws.name} className="sidebar-workspace">
              <div className="sidebar-workspace-name">{ws.name}</div>
              {ws.active_node && (
                <div className="sidebar-active-node">
                  {ws.active_node.title}
                  {tqdm?.total != null && ` (${tqdm.n}/${tqdm.total})`}
                </div>
              )}
              <progress
                className={`progress progress-${ws.status === "active" ? "primary" : "neutral"} w-full`}
                value={progress * 100}
                max={100}
              />
              <div className="sidebar-workspace-meta">
                {ws.boxes_total > 0 && `${ws.boxes_done}/${ws.boxes_total}`}
                {timeLeft(ws) && <span> {timeLeft(ws)}</span>}
              </div>
              <div className="sidebar-workspace-actions">
                {ws.paused ? (
                  <button
                    className="btn btn-sm"
                    title="Resume"
                    type="button"
                    onClick={() => pauseWorkspace(roomName, false)}
                  >
                    <Play />
                  </button>
                ) : (
                  <button
                    className="btn btn-sm"
                    title="Pause"
                    type="button"
                    onClick={() => pauseWorkspace(roomName, true)}
                  >
                    <Pause />
                  </button>
                )}
                <button
                  className="btn btn-sm"
                  title="Stop"
                  type="button"
                  onClick={() => stopWorkspace(roomName)}
                >
                  <Stop />
                </button>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
