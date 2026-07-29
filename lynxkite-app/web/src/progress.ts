export function formatWorkspaceEta(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds <= 0) return "done";
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  if (minutes > 0) return `~${minutes}m ${remainingSeconds}s left`;
  return `~${remainingSeconds}s left`;
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
  const boxesTotal = Number(workspace?.boxes_total ?? 0);
  const fraction = Math.max(0, Math.min(1, Number(workspace?.progress_fraction ?? 0)));

  return {
    activeNode: workspace?.active_node ?? null,
    boxesDone,
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
