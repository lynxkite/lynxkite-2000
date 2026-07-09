"""A Deep Agents backend that represents a workspace as a file."""

import json
import pathlib
import asyncio
from typing import Any
from deepagents.backends import protocol, state
from lynxkite_core import ops, workspace
from . import python_workspace_conversion
from . import sync_workspaces
from . import instructions

try:
    from lynxkite_app import crdt
except ImportError:
    crdt = None  # type: ignore


class WorkspaceBackend(state.StateBackend):
    def __init__(self, workspace: str) -> None:
        super().__init__()
        self._workspace = workspace

    def _read_files(self) -> dict[str, Any]:
        return {
            "/workspace.py": {
                "content": get_workspace_file_content(self._workspace),
                "modified_at": "",
            },
            "/boxes.py": {
                "content": get_boxes_file_content(self._workspace),
                "modified_at": "",
            },
            "/errors.txt": {
                "content": get_errors_file_content(self._workspace),
                "modified_at": "",
            },
            "/layout.json": {
                "content": get_workspace_layout(self._workspace),
                "modified_at": "",
            },
        }

    def _send_files_update(self, update: dict[str, Any]) -> None:
        if "/boxes.py" in update:
            set_boxes_file_content(self._workspace, update["/boxes.py"]["content"])
        if "/workspace.py" in update:
            set_workspace_file_content(
                self._workspace, update["/workspace.py"]["content"]
            )
        if "/layout.json" in update:
            set_layout_file_content(
                self._workspace, json.loads(update["/layout.json"]["content"])
            )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> protocol.EditResult:
        # We convert workspace.py to a Workspace on edit. This can fail in many ways. Tell the agent.
        try:
            return super().edit(file_path, old_string, new_string, replace_all)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return protocol.EditResult(error=str(e))


def get_workspace_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    return python_workspace_conversion.workspace_to_python(ws)


def set_workspace_file_content(ws_path: str, content: str) -> None:
    old_ws = workspace.Workspace.load(ws_path)
    ops.load_user_scripts(ws_path)
    ws = python_workspace_conversion.python_to_workspace(content)
    ws.env = old_ws.env
    ws.assistant_messages = old_ws.assistant_messages
    ws.paused = old_ws.paused
    sync_workspaces.update_node_ids(source=ws, target=old_ws)
    sync_workspaces.update_ws_positions(source=old_ws, target=ws)
    if not ws.paused:
        asyncio.run(ops.EXECUTORS[ws.env](ws, ops.CATALOGS[ws.env]))
    ws.save(ws_path)
    if crdt:
        room = crdt.get_room_or_none(ws_path)
        if room is not None:
            crdt.update_workspace(room.ws, ws)


def get_boxes_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    p = pathlib.Path(ws_path).parent / "boxes.py"
    if not p.exists():
        module_name = (
            f"{ops.to_python_module_name(p.parent)}.boxes"
            if str(p.parent) != "."
            else "boxes"
        )
        return instructions.get_boxes_prompt(ws.env, module_name)
    with open(p) as f:
        return f.read()


def set_boxes_file_content(ws_path: str, content: str) -> None:
    p = pathlib.Path(ws_path).parent / "boxes.py"
    with open(p, "w") as f:
        f.write(content)


def get_errors_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    errors = ((n.id, n.data.error) for n in ws.nodes if n.data.error)
    return "\n---\n".join(
        f"An error occured in {node_id}:\n {error}" for node_id, error in errors
    )


def get_workspace_layout(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    layout = {
        "nodes": [
            {
                "id": node.id,
                "position": {"x": node.position.x, "y": node.position.y},
                "width": node.width,
                "height": node.height,
                "collapsed": node.data.collapsed,
            }
            for node in ws.nodes
        ]
    }
    return json.dumps(layout)


def set_layout_file_content(ws_path: str, layout: dict[str, Any]) -> None:
    ws = workspace.Workspace.load(ws_path)
    node_by_id = {node.id: node for node in ws.nodes}
    for node_layout in layout.get("nodes", []):
        node_id = node_layout.get("id")
        if not node_id or node_id not in node_by_id:
            continue
        node = node_by_id[node_id]
        position = node_layout.get("position", {})
        if "x" in position and "y" in position:
            node.position = workspace.Position(x=position["x"], y=position["y"])
        if "width" in node_layout:
            node.width = node_layout["width"]
        if "height" in node_layout:
            node.height = node_layout["height"]
        if "collapsed" in node_layout:
            node.data.collapsed = bool(node_layout["collapsed"])
    ws.save(ws_path)
    if crdt:
        room = crdt.get_room_or_none(ws_path)
        if room is not None:
            crdt.update_workspace(room.ws, ws)
