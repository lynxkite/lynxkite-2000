import {
  formatWorkspaceProgressSuffix,
  getWorkspaceProgress,
  workspaceProgressColor,
} from "../progress";
import { useWorkspaceProgress } from "./useWorkspaceProgress.ts";

export function WorkspaceProgress({
  path,
  enabled,
}: {
  path: string | undefined;
  enabled?: boolean;
}) {
  const workspace = useWorkspaceProgress(path, enabled);
  if (!workspace) {
    return null;
  }
  const progress = getWorkspaceProgress(workspace);
  if (progress.boxesTotal <= 0 || progress.status === "idle") {
    return null;
  }
  const suffix = formatWorkspaceProgressSuffix(progress);
  const label = progress.activeNode?.title || "Workspace progress";
  const meta = [
    `${progress.percent.toFixed(0)}%`,
    `${progress.boxesDone}/${progress.boxesTotal}`,
    suffix,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="workspace-progress-compact" title={`${label} — ${meta}`}>
      <progress
        className={`progress progress-${workspaceProgressColor(progress.status)}`}
        value={progress.percent}
        max={100}
      />
      <div className="workspace-progress-compact-meta">{meta}</div>
    </div>
  );
}
