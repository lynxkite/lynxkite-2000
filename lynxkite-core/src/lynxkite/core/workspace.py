"""For working with LynxKite workspaces."""

import json
from typing import Optional
import dataclasses
import enum
import os
import pycrdt
import pydantic
import tempfile
from . import ops


class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="allow",
    )


class Position(BaseConfig):
    x: float
    y: float


class NodeStatus(str, enum.Enum):
    planned = "planned"
    active = "active"
    done = "done"


class WorkspaceNodeData(BaseConfig):
    title: str
    params: dict
    display: Optional[object] = None
    error: Optional[str] = None
    status: NodeStatus = NodeStatus.done
    # Also contains a "meta" field when going out.
    # This is ignored when coming back from the frontend.


class WorkspaceNode(BaseConfig):
    # The naming of these attributes matches the ones for the NodeBase type in React flow
    # modyfing them will break the frontend.
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position
    _crdt: pycrdt.Map

    def publish_started(self):
        """Notifies the frontend that work has started on this node."""
        self.data.error = None
        self.data.status = NodeStatus.active
        if hasattr(self, "_crdt"):
            with self._crdt.doc.transaction():
                self._crdt["data"]["error"] = None
                self._crdt["data"]["status"] = NodeStatus.active

    def publish_result(self, result: ops.Result):
        """Sends the result to the frontend. Call this in an executor when the result is available."""
        self.data.display = result.display
        self.data.error = result.error
        self.data.status = NodeStatus.done
        if hasattr(self, "_crdt"):
            with self._crdt.doc.transaction():
                self._crdt["data"]["display"] = result.display
                self._crdt["data"]["error"] = result.error
                self._crdt["data"]["status"] = NodeStatus.done

    def publish_error(self, error: Exception | str | None):
        """Can be called with None to clear the error state."""
        result = ops.Result(error=str(error) if error else None)
        self.publish_result(result)


class WorkspaceEdge(BaseConfig):
    id: str
    source: str
    target: str
    sourceHandle: str
    targetHandle: str


class Workspace(BaseConfig):
    """A workspace is a representation of a computational graph that consists of nodes and edges.

    Each node represents an operation or task, and the edges represent the flow of data between
    the nodes. Each workspace is associated with an environment, which determines the operations
    that can be performed in the workspace and the execution method for the operations.
    """

    env: str = ""
    nodes: list[WorkspaceNode] = dataclasses.field(default_factory=list)
    edges: list[WorkspaceEdge] = dataclasses.field(default_factory=list)
    _crdt: pycrdt.Map


async def execute(ws: Workspace):
    if ws.env in ops.EXECUTORS:
        await ops.EXECUTORS[ws.env](ws)


def save(ws: Workspace, path: str):
    """Persist a workspace to a local file in JSON format."""
    j = ws.model_dump()
    j = json.dumps(j, indent=2, sort_keys=True) + "\n"
    dirname, basename = os.path.split(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    # Create temp file in the same directory to make sure it's on the same filesystem.
    with tempfile.NamedTemporaryFile(
        "w", prefix=f".{basename}.", dir=dirname, delete=False
    ) as f:
        temp_name = f.name
        f.write(j)
    os.replace(temp_name, path)


def load(path: str) -> Workspace:
    """Load a workspace from a file.

    After loading the workspace, the metadata of the workspace is updated.

    Args:
        path (str): The path to the file to load the workspace from.

    Returns:
        Workspace: The loaded workspace object, with updated metadata.
    """
    with open(path) as f:
        j = f.read()
    ws = Workspace.model_validate_json(j)
    # Metadata is added after loading. This way code changes take effect on old boxes too.
    _update_metadata(ws)
    return ws


def _update_metadata(ws: Workspace) -> Workspace:
    """Update the metadata of the given workspace object.

    The metadata is the information about the operations that the nodes in the workspace represent,
    like the parameters and their possible values.
    This information comes from the catalog of operations for the environment of the workspace.

    Args:
        ws: The workspace object to update.

    Returns:
        Workspace: The updated workspace object.
    """
    catalog = ops.CATALOGS.get(ws.env, {})
    nodes = {node.id: node for node in ws.nodes}
    done = set()
    while len(done) < len(nodes):
        for node in ws.nodes:
            if node.id in done:
                # TODO: Can nodes with the same ID reference different operations?
                continue
            data = node.data
            op = catalog.get(data.title)
            if op:
                data.meta = op
                node.type = op.type
                if data.error == "Unknown operation.":
                    data.error = None
            else:
                data.error = "Unknown operation."
            done.add(node.id)
    return ws
