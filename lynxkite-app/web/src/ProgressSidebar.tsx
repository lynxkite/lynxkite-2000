// Compact workspace progress list embedded in the directory sidebar.
import { useEffect, useState } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import Pause from "~icons/tabler/player-pause-filled";
import Play from "~icons/tabler/player-play-filled";
import Stop from "~icons/tabler/player-stop-filled";
import {
  formatWorkspaceProgressSuffix,
  getWorkspaceProgress,
  parseProgressWorkspace,
  pauseWorkspace,
  stopWorkspace,
  workspaceProgressColor,
} from "./progress";

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
        const parsed = parseProgressWorkspace(value);
        if (parsed) ws.push(parsed);
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

  return (
    <div className="sidebar-progress-section">
      <div className="sidebar-section-title">Running workspaces</div>
      {workspaces.length === 0 ? (
        <p className="sidebar-empty">No workspaces running.</p>
      ) : (
        workspaces.map((ws) => {
          const roomName = ws.room_name || ws.name;
          const progress = getWorkspaceProgress(ws);
          const tqdm = progress.activeNode?.tqdm;
          const suffix = formatWorkspaceProgressSuffix(progress);
          return (
            <div key={ws.name} className="sidebar-workspace">
              <div className="sidebar-workspace-name">{ws.name}</div>
              {progress.activeNode && (
                <div className="sidebar-active-node">
                  {progress.activeNode.title}
                  {tqdm?.total != null && ` (${tqdm.n}/${tqdm.total})`}
                </div>
              )}
              <progress
                className={`progress progress-${workspaceProgressColor(progress.status)} w-full`}
                value={progress.percent}
                max={100}
              />
              <div className="sidebar-workspace-meta">
                {progress.boxesTotal > 0 && `${progress.boxesDone}/${progress.boxesTotal}`}
                {suffix && <span> {suffix}</span>}
              </div>
              <div className="sidebar-workspace-actions">
                {progress.paused ? (
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
