import { formatWorkspaceEta, getWorkspaceProgress } from "../progress";
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
  const etaText = formatWorkspaceEta(progress.etaSeconds);
  const label = progress.activeNode?.title || "Workspace progress";
  const metaParts = [
    `${progress.percent.toFixed(0)}%`,
    `${progress.boxesDone}/${progress.boxesTotal}`,
    etaText,
  ].filter(Boolean);

  return (
    <div className="workspace-progress-compact" title={`${label} — ${metaParts.join(" ")}`}>
      <progress
        className={`progress progress-${progress.status === "active" ? "primary" : "neutral"}`}
        value={progress.percent}
        max={100}
      />
      <div className="workspace-progress-compact-meta">{metaParts.join(" ")}</div>
    </div>
  );
}
