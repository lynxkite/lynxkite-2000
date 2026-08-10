import { useEffect, useState } from "react";
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
  // Re-render once per second so ETA can tick locally between backend updates.
  const [, setTick] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  if (!workspace) {
    return null;
  }
  const progress = getWorkspaceProgress(workspace, Date.now());
  if (progress.boxesTotal <= 0 || progress.status === "idle") {
    return null;
  }
  const etaText = formatWorkspaceEta(progress.displayEtaSeconds);
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
