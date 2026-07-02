"""A Deep Agents backend that represents a workspace as a file."""

import json
import pathlib
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

To use them in the workspace, call them in `workspace.py` with this custom module name: MODULE_NAME
For example:
    MODULE_NAME.blur(...)
    MODULE_NAME.read_csv(...)
"""
from lynxkite_core import ops
op = ops.op_registration(ENV) # DO NOT CHANGE THIS LINE!

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


def _replace_node_id_and_edges(
    ws: workspace.Workspace, old_id: str, new_id: str
) -> None:
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
    # used for sorting source nodes to match the order of target nodes as much as possible (important for testing)
    new_id_old_order = {sn.id: len(target.nodes) for sn in source.nodes}

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
            new_id_old_order[source_node.id] = target.nodes.index(target_node)
            _replace_node_id_and_edges(target, old_target_id, source_node.id)
            assigned_source.add(source_idx)
            assigned_target.add(target_idx)
            return True
        return False

    while len(assigned_target) < len(target.nodes) and len(assigned_source) < len(
        source.nodes
    ):
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
        source_neighbors = {
            node.id: _get_node_neighbors(source, node.id) for node in source.nodes
        }
        target_neighbors = {
            node.id: _get_node_neighbors(target, node.id) for node in target.nodes
        }

        def match_by_neighbors_lenient(s_neighbors, t_neighbors) -> bool:
            if s_neighbors == t_neighbors:
                return True
            if s_neighbors is None or t_neighbors is None:
                return False
            # some ids of neighbors may not have been matched yet, so we only check that the lengths of the neighbor sets are the same and that there is at least one matching neighbor in each set.
            inp1, out1 = s_neighbors
            inp2, out2 = t_neighbors
            return (
                len(inp1) == len(inp2)
                and len(out1) == len(out2)
                and (
                    inp1.intersection(inp2) != set() or out1.intersection(out2) != set()
                )
            )

        assigned = try_match_by(
            lambda s, t: s.data.op_id == t.data.op_id
            and s.data.params == t.data.params
            and match_by_neighbors_lenient(
                source_neighbors.get(s.id), target_neighbors.get(t.id)
            )
            if s.type != "comment" and t.type != "comment"
            # for coments, match based on text without whitespace and line breaks
            else s.data.params.get("text", "").replace(" ", "").replace("\n", "")
            == t.data.params.get("text", "").replace(" ", "").replace("\n", "")
        )
        if assigned:
            continue
        # match strictly by neighbors, including ids of neighbors
        assigned = try_match_by(
            lambda s, t: s.data.op_id == t.data.op_id
            and s.data.params == t.data.params
            and source_neighbors.get(s.id) == target_neighbors.get(t.id)
        )
        if assigned:
            continue
        source.nodes.sort(key=lambda n: new_id_old_order.get(n.id, len(source.nodes)))
        return  # Give up if we can't find any more matches.
    source.nodes.sort(key=lambda n: new_id_old_order.get(n.id, len(source.nodes)))


def _update_ws_positions(
    source: workspace.Workspace, target: workspace.Workspace
) -> None:
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
    # For new non-comment nodes, make up a new position based on the positions of their neighbors.
    for node in target.nodes:
        source_node = source_nodes_by_id.get(node.id)
        if source_node is not None or node.type == "comment":
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

    # For new comments, place them above the box defined in the next available line.
    # We assume that the comment's ID is of the form "comment on line N" where N is an integer.
    def _intd(s: str) -> int | None:
        try:
            return int(s)
        except ValueError:
            return None

    node_by_line_id = {
        _intd(node.id.split()[-1]): node
        for node in target.nodes
        if _intd(node.id.split()[-1]) is not None
    }
    for node in target.nodes:
        source_node = source_nodes_by_id.get(node.id)
        if source_node is not None or node.type != "comment":
            continue
        line_id = _intd(node.id.split()[-1])
        next_line_id = min(
            filter(lambda x: x > line_id, node_by_line_id.keys()), default=line_id
        )
        next_line_node = node_by_line_id.get(next_line_id)
        if next_line_node is None:
            continue
        node.position = workspace.Position(
            x=next_line_node.position.x, y=next_line_node.position.y - 50
        )
        continue


def set_workspace_file_content(ws_path: str, content: str) -> None:
    old_ws = workspace.Workspace.load(ws_path)
    ops.load_user_scripts(ws_path)
    ws = python_workspace_conversion.python_to_workspace(content)
    ws.env = old_ws.env
    _update_node_ids(source=ws, target=old_ws)
    _update_ws_positions(source=old_ws, target=ws)
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
        return BOXES_PLACEHOLDER.replace("ENV", f'"{ws.env}"').replace(
            "MODULE_NAME", module_name
        )
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
