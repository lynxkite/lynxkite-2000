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

function parseEtaTimestampMs(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    // Accept either unix seconds or milliseconds.
    return value > 1e12 ? value : value * 1000;
  }
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

function getBackendEtaAnchorMs(workspace: any): number | null {
  const candidates = [
    workspace?.eta_updated_at,
    workspace?.eta_updated_at_ms,
    workspace?.progress_updated_at,
    workspace?.updated_at,
  ];
  for (const candidate of candidates) {
    const parsed = parseEtaTimestampMs(candidate);
    if (parsed != null) return parsed;
  }
  return null;
}

export function attachWorkspaceEtaAnchor(workspace: any, previous: any, nowMs = Date.now()) {
  if (!workspace || typeof workspace !== "object") {
    return workspace;
  }

  const etaSeconds = typeof workspace.eta_seconds === "number" ? workspace.eta_seconds : null;
  const previousEtaSeconds =
    typeof previous?.eta_seconds === "number" ? previous.eta_seconds : null;
  const backendAnchorMs = getBackendEtaAnchorMs(workspace);

  let etaAnchorMs = nowMs;
  let etaAnchorSeconds = etaSeconds;

  if (backendAnchorMs != null) {
    etaAnchorMs = backendAnchorMs;
  } else if (
    previous &&
    previousEtaSeconds != null &&
    etaSeconds != null &&
    previousEtaSeconds === etaSeconds &&
    previous?.status === workspace?.status &&
    previous?.active_node?.id === workspace?.active_node?.id
  ) {
    // Keep the old anchor when backend data is unchanged so ETA can tick locally.
    etaAnchorMs = previous.__eta_anchor_ms ?? nowMs;
    etaAnchorSeconds = previous.__eta_anchor_seconds ?? etaSeconds;
  }

  return {
    ...workspace,
    __eta_anchor_ms: etaAnchorMs,
    __eta_anchor_seconds: etaAnchorSeconds,
  };
}

export function getWorkspaceDisplayEtaSeconds(workspace: any, nowMs = Date.now()): number | null {
  if (workspace?.status === "paused") {
    if (typeof workspace?.__eta_anchor_seconds === "number") {
      return Math.max(0, workspace.__eta_anchor_seconds);
    }
    if (typeof workspace?.eta_seconds === "number") {
      return Math.max(0, workspace.eta_seconds);
    }
    return null;
  }
  const anchorSeconds =
    typeof workspace?.__eta_anchor_seconds === "number"
      ? workspace.__eta_anchor_seconds
      : typeof workspace?.eta_seconds === "number"
        ? workspace.eta_seconds
        : null;
  if (anchorSeconds == null) {
    return null;
  }
  const anchorMs =
    typeof workspace?.__eta_anchor_ms === "number" ? workspace.__eta_anchor_ms : nowMs;
  const elapsed = Math.max(0, (nowMs - anchorMs) / 1000);
  return Math.max(0, anchorSeconds - elapsed);
}

/** Normalize backend progress payload for UI components. */
export function getWorkspaceProgress(workspace: any, nowMs = Date.now()) {
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
    displayEtaSeconds: getWorkspaceDisplayEtaSeconds(workspace, nowMs),
    etaSeconds: typeof workspace?.eta_seconds === "number" ? workspace.eta_seconds : null,
  };
}
