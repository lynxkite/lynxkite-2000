"""Workspace-level progress aggregation.

``progress_fraction`` counts completed boxes plus partial credit from the active
box's tqdm (``n / total``), divided by total progress boxes.

``eta_seconds`` uses ``elapsed * (1 - f) / f`` when elapsed time and fraction
are known; otherwise extrapolates from the active box's tqdm rate.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from .workspace import NodeStatus, Workspace, WorkspaceNode


def is_progress_box(node: WorkspaceNode) -> bool:
    return node.type != "node_group" and node.data.op_id != "Comment"


def workspace_display_name(room_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(room_name).name.removesuffix(".lynxkite.json"))


def _active_node_info(
    nodes: list[WorkspaceNode],
) -> tuple[dict[str, Any] | None, float | None, float]:
    for node in nodes:
        if node.data.status != NodeStatus.active:
            continue
        t = node.data.telemetry or {}
        n, total, rate = t.get("n"), t.get("total"), t.get("rate")
        tqdm = {"n": n, "total": total} if isinstance(total, (int, float)) and total > 0 else None
        partial = 0.0
        box_eta = None
        if isinstance(n, (int, float)) and isinstance(total, (int, float)) and total > 0:
            partial = max(0.0, min(1.0, n / total))
            if isinstance(rate, (int, float)) and rate > 0:
                box_eta = max(0.0, (total - n) / rate)
        return {"id": node.id, "title": node.data.title, "tqdm": tqdm}, box_eta, partial
    return None, None, 0.0


def compute_workspace_eta_seconds(
    *,
    progress_fraction: float,
    elapsed_seconds: float | None,
    active_box_eta: float | None = None,
    boxes_done: int = 0,
    boxes_total: int = 0,
) -> float | None:
    if boxes_total > 0 and boxes_done >= boxes_total:
        return 0.0
    if elapsed_seconds and 0 < progress_fraction < 1:
        return max(0.0, elapsed_seconds * (1.0 - progress_fraction) / progress_fraction)
    if active_box_eta is not None:
        return active_box_eta + max(0, boxes_total - boxes_done - 1) * active_box_eta
    return None


def compute_workspace_progress(
    ws: Workspace,
    *,
    room_name: str,
    elapsed_seconds: float | None = None,
    gpus: int | None = None,
) -> dict[str, Any]:
    nodes = [n for n in (ws.nodes or []) if is_progress_box(n)]
    boxes_total = len(nodes)
    boxes_done = sum(n.data.status == NodeStatus.done for n in nodes)
    active_count = sum(n.data.status == NodeStatus.active for n in nodes)
    paused = bool(ws.paused)
    active_node, active_box_eta, partial = _active_node_info(nodes)
    fraction = max(0.0, min(1.0, (boxes_done + partial) / boxes_total)) if boxes_total else 0.0
    if not boxes_total:
        status = "idle"
    elif boxes_done == boxes_total:
        status = "done"
    elif paused:
        status = "paused"
    else:
        status = "active" if active_count else "running"
    return {
        "name": workspace_display_name(room_name),
        "room_name": room_name,
        "status": status,
        "boxes_done": boxes_done,
        "boxes_total": boxes_total,
        "active_node": active_node,
        "progress_fraction": fraction,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": compute_workspace_eta_seconds(
            progress_fraction=fraction,
            elapsed_seconds=elapsed_seconds,
            active_box_eta=active_box_eta,
            boxes_done=boxes_done,
            boxes_total=boxes_total,
        ),
        "gpus": gpus if gpus is not None else (ws.execution_options or {}).get("gpus", 0),
        "paused": paused,
    }
