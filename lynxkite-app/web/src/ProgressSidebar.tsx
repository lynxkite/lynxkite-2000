// Compact workspace progress list embedded in the directory sidebar.
import Pause from "~icons/tabler/player-pause-filled";
import Play from "~icons/tabler/player-play-filled";
import Stop from "~icons/tabler/player-stop-filled";
import {
  formatWorkspaceProgressSuffix,
  getWorkspaceProgress,
  pauseWorkspace,
  stopWorkspace,
  useProgress,
  workspaceProgressColor,
} from "./progress";

export function WorkspaceProgressList() {
  const { workspaces } = useProgress(true);

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
