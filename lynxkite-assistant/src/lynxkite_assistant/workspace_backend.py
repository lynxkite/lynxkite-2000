"""A Deep Agents backend that represents a workspace as a file."""

import pathlib
from pprint import pprint
from typing import Any, Callable
from deepagents.backends import protocol, state
from lynxkite_core import ops, workspace
from . import python_workspace_conversion

try:
    from lynxkite_app import crdt
except ImportError:
    crdt = None  # type: ignore

BOXES_PLACEHOLDER = '''
"""Custom box definitions for the workspace.

To add a custom box, define a function here and decorate it with @op.
The positional arguments of the function become its inputs, and the keyword-only arguments become its parameters.
E.g.:

    @op("Blur")
    def blur(image: Image.Image, *, radius = 5):
        return image.filter(ImageFilter.GaussianBlur(radius))

    @op("Read CSV")
    def read_csv(*, path: str):
        return pd.read_csv(path)

"""
from lynxkite_core import ops
op = ops.op_registration(ENV)

# Add new box definitions here.
'''.strip()


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
            "/boxes.py": {"content": get_boxes_file_content(self._workspace), "modified_at": ""},
        }

    def _send_files_update(self, update: dict[str, Any]) -> None:
        print("update")
        pprint(update)
        if "/boxes.py" in update:
            set_boxes_file_content(self._workspace, update["/boxes.py"]["content"])
        if "/workspace.py" in update:
            set_workspace_file_content(self._workspace, update["/workspace.py"]["content"])

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


def _get_node_neighbors(
    ws: workspace.Workspace,
    node_id: str,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    inputs = {
        (edge.source, edge.sourceHandle, edge.targetHandle)
        for edge in ws.edges
        if edge.target == node_id
    }
    outputs = {
        (edge.target, edge.sourceHandle, edge.targetHandle)
        for edge in ws.edges
        if edge.source == node_id
    }
    return inputs, outputs


def _replace_node_id_and_edges(ws: workspace.Workspace, old_id: str, new_id: str) -> None:
    if old_id == new_id:
        return
    for node in ws.nodes:
        if node.id == old_id:
            node.id = new_id
            break
    for edge in ws.edges:
        if edge.source == old_id:
            edge.source = new_id
        if edge.target == old_id:
            edge.target = new_id


def _update_node_ids(source: workspace.Workspace, target: workspace.Workspace) -> None:
    """Update node IDs in target to match those in source as much as possible."""
    assigned_source: set[int] = set()
    assigned_target: set[int] = set()

    def try_match_by(
        match: Callable[[workspace.WorkspaceNode, workspace.WorkspaceNode], bool],
    ) -> bool:
        for target_idx, target_node in enumerate(target.nodes):
            if target_idx in assigned_target:
                continue
            unique = True
            for target2_idx, target2_node in enumerate(target.nodes):
                if target2_idx == target_idx or target2_idx in assigned_target:
                    continue
                if match(target2_node, target_node):
                    unique = False
            if not unique:
                continue
            candidates = []
            for source_idx, source_node in enumerate(source.nodes):
                if source_idx in assigned_source:
                    continue
                if match(source_node, target_node):
                    candidates.append(source_idx)
            if len(candidates) != 1:
                continue
            source_idx = candidates[0]
            source_node = source.nodes[source_idx]
            old_target_id = target_node.id
            _replace_node_id_and_edges(target, old_target_id, source_node.id)
            assigned_source.add(source_idx)
            assigned_target.add(target_idx)
            return True
        return False

    while len(assigned_target) < len(target.nodes) and len(assigned_source) < len(source.nodes):
        # Try to match by op ID only.
        assigned = try_match_by(lambda s, t: s.data.op_id == t.data.op_id)
        if assigned:
            continue
        # Try to match by op ID and params.
        assigned = try_match_by(
            lambda s, t: s.data.op_id == t.data.op_id and s.data.params == t.data.params
        )
        if assigned:
            continue
        # Try to match by op ID, params, and neighbors.
        source_neighbors = {node.id: _get_node_neighbors(source, node.id) for node in source.nodes}
        target_neighbors = {node.id: _get_node_neighbors(target, node.id) for node in target.nodes}
        assigned = try_match_by(
            lambda s, t: s.data.op_id == t.data.op_id
            and s.data.params == t.data.params
            and source_neighbors.get(s.id) == target_neighbors.get(t.id)
        )
        if assigned:
            continue
        return  # Give up if we can't find any more matches.


def _update_ws_positions(source: workspace.Workspace, target: workspace.Workspace) -> None:
    """Update node positions in target by node ID."""
    source_nodes_by_id = {node.id: node for node in source.nodes}
    # Copy the dimensions of existing nodes.
    for node in target.nodes:
        source_node = source_nodes_by_id.get(node.id)
        if source_node is None:
            continue
        node.position = source_node.position
        node.width = source_node.width
        node.height = source_node.height
        node.data.collapsed = source_node.data.collapsed
    # For new nodes, make up a new position based on the positions of their neighbors.
    for node in target.nodes:
        source_node = source_nodes_by_id.get(node.id)
        if source_node is not None:
            continue
        inputs, outputs = _get_node_neighbors(target, node.id)
        x = 0
        y = 0
        count = 0
        for neighbors, x_offset in [(inputs, 500), (outputs, -500)]:
            for neighbor_id, _, _ in neighbors:
                neighbor_source_node = source_nodes_by_id.get(neighbor_id)
                if neighbor_source_node is None:
                    continue
                x += neighbor_source_node.position.x + x_offset
                y += neighbor_source_node.position.y
                count += 1
        if count > 0:
            x /= count
            y /= count
            node.position = workspace.Position(x=x, y=y)


def set_workspace_file_content(ws_path: str, content: str) -> None:
    old_ws = workspace.Workspace.load(ws_path)
    ops.load_user_scripts(ws_path)
    ws = python_workspace_conversion.python_to_workspace(content)
    ws.env = old_ws.env
    _update_node_ids(source=ws, target=old_ws)
    _update_ws_positions(source=old_ws, target=ws)
    ws.save(ws_path)
    if crdt:
        room = crdt.get_room_if_exists(ws_path)
        crdt.update_workspace(room.ws, ws)


def get_boxes_file_content(ws_path: str) -> str:
    ws = workspace.Workspace.load(ws_path)
    p = pathlib.Path(ws_path).parent / "boxes.py"
    if not p.exists():
        return BOXES_PLACEHOLDER.replace("ENV", f'"{ws.env}"')
    with open(p) as f:
        return f.read()


def set_boxes_file_content(ws_path: str, content: str) -> None:
    p = pathlib.Path(ws_path).parent / "boxes.py"
    with open(p, "w") as f:
        f.write(content)
