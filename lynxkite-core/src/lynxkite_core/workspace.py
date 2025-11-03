"""For working with LynxKite workspaces."""

import json
import pathlib
from typing import Any, Optional, TYPE_CHECKING
import dataclasses
import enum
import os
import pydantic
import tempfile
from . import ops

if TYPE_CHECKING:
    import pycrdt
    import fastapi
    from lynxkite_core import ops


class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="allow",
    )


class Position(BaseConfig):
    x: float
    y: float


class NodeStatus(enum.StrEnum):
    planned = "planned"
    active = "active"
    done = "done"


class WorkspaceNodeData(BaseConfig):
    title: str
    op_id: str
    params: dict
    display: Optional[Any] = None
    input_metadata: Optional[list[dict]] = None
    error: Optional[str] = None
    collapsed: Optional[bool] = None
    status: NodeStatus = NodeStatus.done
    meta: Optional["ops.Op"] = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def fill_op_id_if_missing(cls, data: dict) -> dict:
        """Compatibility with old workspaces that don't have op_id."""
        if "op_id" not in data:
            data["op_id"] = data["title"]
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def ignore_meta(cls, data: dict) -> dict:
        """Metadata is never loaded. We will use fresh metadata."""
        data["meta"] = None
        return data


class WorkspaceNode(BaseConfig):
    # Most of these fields are shared with ReactFlow.
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position
    width: Optional[float] = None
    height: Optional[float] = None
    _crdt: Optional["pycrdt.Map"] = None

    def publish_started(self):
        """Notifies the frontend that work has started on this node."""
        self.data.error = None
        self.data.status = NodeStatus.active
        if self._crdt and "data" in self._crdt:
            with self._crdt.doc.transaction():
                self._crdt["data"]["error"] = None
                self._crdt["data"]["status"] = NodeStatus.active

    def publish_result(self, result: ops.Result):
        """Sends the result to the frontend. Call this in an executor when the result is available."""
        self.data.display = result.display
        self.data.input_metadata = result.input_metadata
        self.data.error = result.error
        self.data.status = NodeStatus.done
        if self._crdt and "data" in self._crdt:
            with self._crdt.doc.transaction():
                try:
                    self._crdt["data"]["status"] = NodeStatus.done
                    self._crdt["data"]["display"] = self.data.display
                    self._crdt["data"]["input_metadata"] = self.data.input_metadata
                    self._crdt["data"]["error"] = self.data.error
                except Exception as e:
                    self._crdt["data"]["error"] = str(e)
                    raise e

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


@dataclasses.dataclass
class WorkspaceExecutionContext:
    """Context passed to ops during execution."""

    app: "fastapi.FastAPI | None"


class Workspace(BaseConfig):
    """A workspace is a representation of a computational graph that consists of nodes and edges.

    Each node represents an operation or task, and the edges represent the flow of data between
    the nodes. Each workspace is associated with an environment, which determines the operations
    that can be performed in the workspace and the execution method for the operations.
    """

    env: str = ""
    nodes: list[WorkspaceNode] = dataclasses.field(default_factory=list)
    edges: list[WorkspaceEdge] = dataclasses.field(default_factory=list)
    path: Optional[str] = None
    _crdt: Optional["pycrdt.Map"] = None

    def normalize(self):
        if self.env not in ops.CATALOGS:
            return self
        catalog = ops.CATALOGS[self.env]
        _ops = {n.id: catalog[n.data.op_id] for n in self.nodes if n.data.op_id in catalog}
        valid_targets = set()
        valid_sources = set()
        for n in self.nodes:
            if n.id in _ops:
                for h in _ops[n.id].inputs:
                    valid_targets.add((n.id, h.name))
                for h in _ops[n.id].outputs:
                    valid_sources.add((n.id, h.name))
        self.edges = [
            edge
            for edge in self.edges
            if (edge.source, edge.sourceHandle) in valid_sources
            and (edge.target, edge.targetHandle) in valid_targets
        ]

    def has_executor(self):
        return self.env in ops.EXECUTORS

    async def execute(self, ctx: WorkspaceExecutionContext | None = None):
        return await ops.EXECUTORS[self.env](self, ctx)

    def model_dump_json(self) -> str:
        """Returns the workspace as JSON."""
        # Pydantic can't sort the keys. TODO: Keep an eye on https://github.com/pydantic/pydantic-core/pull/1637.
        j = self.model_dump()
        if "path" in j:
            del j["path"]
        j = json.dumps(j, indent=2, sort_keys=True) + "\n"
        return j

    def save(self, path: str | pathlib.Path):
        """Persist the workspace to a local file in JSON format."""
        path = str(path)
        j = self.model_dump_json()
        dirname, basename = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        # Create temp file in the same directory to make sure it's on the same filesystem.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=f".{basename}.", dir=dirname, delete=False
        ) as f:
            temp_name = f.name
            f.write(j)
        os.replace(temp_name, path)

    @staticmethod
    def load(path: str | pathlib.Path) -> "Workspace":
        """Load a workspace from a file.

        After loading the workspace, the metadata of the workspace is updated.

        Args:
            path: The path to the file to load the workspace from.

        Returns:
            Workspace: The loaded workspace object, with updated metadata.
        """
        path = str(path)
        with open(path, encoding="utf-8") as f:
            j = f.read()
        ws = Workspace.model_validate_json(j)
        # Metadata is added after loading. This way code changes take effect on old boxes too.
        ws.update_metadata()
        ws.path = path
        return ws

    def update_metadata(self):
        """Update the metadata of this workspace.

        The metadata is the information about the operations that the nodes in the workspace represent,
        like the parameters and their possible values.
        This information comes from the catalog of operations for the environment of the workspace.
        """
        if self.env not in ops.CATALOGS:
            return self
        catalog = ops.CATALOGS[self.env]
        for node in self.nodes:
            data = node.data
            op = catalog.get(data.op_id)
            if op:
                if getattr(data, "meta", None) != op:
                    data.meta = op
                    # If the node is connected to a CRDT, update that too.
                    if node._crdt:
                        node._crdt["data"]["meta"] = op.model_dump()
                if node.type != op.type:
                    node.type = op.type
                    if node._crdt:
                        node._crdt["type"] = op.type
                if data.error == "Unknown operation.":
                    data.error = None
                    if node._crdt:
                        node._crdt["data"]["error"] = None
            else:
                data.error = "Unknown operation."
                if node._crdt:
                    node._crdt["data"]["meta"] = {}
                    node._crdt["data"]["error"] = "Unknown operation."

    def connect_crdt(self, ws_crdt: "pycrdt.Map"):
        import pycrdt

        self._crdt = ws_crdt
        with ws_crdt.doc.transaction():
            for nc, np in zip(ws_crdt["nodes"], self.nodes):
                if "data" not in nc:
                    nc["data"] = pycrdt.Map()
                np._crdt = nc

    def add_node(self, func=None, **kwargs):
        """For convenience in e.g. tests."""
        random_string = os.urandom(4).hex()
        if func:
            kwargs["type"] = func.__op__.type
            kwargs["data"] = WorkspaceNodeData(
                title=func.__op__.name, op_id=func.__op__.id, params={}
            )
        elif "title" in kwargs:
            kwargs["data"] = WorkspaceNodeData(
                title=kwargs["title"],
                op_id=kwargs["title"],
                params=kwargs.get("params", {}),
            )
        kwargs.setdefault("type", "basic")
        kwargs.setdefault("id", f"{kwargs['data'].title} {random_string}")
        kwargs.setdefault("position", Position(x=0, y=0))
        kwargs.setdefault("width", 100)
        kwargs.setdefault("height", 100)
        node = WorkspaceNode(**kwargs)  # ty: ignore[missing-argument]
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        source: WorkspaceNode,
        sourceHandle: str,
        target: WorkspaceNode,
        targetHandle: str,
    ):
        """For convenience in e.g. tests."""
        edge = WorkspaceEdge(
            id=f"{source.id} {sourceHandle} to {target.id} {targetHandle}",
            source=source.id,
            target=target.id,
            sourceHandle=sourceHandle,
            targetHandle=targetHandle,
        )
        self.edges.append(edge)
        return edge
