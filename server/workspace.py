'''For working with LynxKite workspaces.'''
from typing import Optional
import dataclasses
import os
import pydantic
import tempfile
import traceback
from . import ops

class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra='allow',
    )

class Position(BaseConfig):
    x: float
    y: float

class WorkspaceNodeData(BaseConfig):
    title: str
    params: dict
    display: Optional[object] = None
    error: Optional[str] = None
    # Also contains a "meta" field when going out.
    # This is ignored when coming back from the frontend.

class WorkspaceNode(BaseConfig):
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position
    parentId: Optional[str] = None

class WorkspaceEdge(BaseConfig):
    id: str
    source: str
    target: str
    sourceHandle: str
    targetHandle: str

class Workspace(BaseConfig):
    env: str = ''
    nodes: list[WorkspaceNode] = dataclasses.field(default_factory=list)
    edges: list[WorkspaceEdge] = dataclasses.field(default_factory=list)


def execute(ws: Workspace):
    if ws.env in ops.EXECUTORS:
        ops.EXECUTORS[ws.env](ws)


def save(ws: Workspace, path: str):
    j = ws.model_dump_json(indent=2)
    dirname, basename = os.path.split(path)
    # Create temp file in the same directory to make sure it's on the same filesystem.
    with tempfile.NamedTemporaryFile('w', prefix=f'.{basename}.', dir=dirname, delete_on_close=False) as f:
        f.write(j)
        f.close()
        os.replace(f.name, path)


def load(path: str):
    with open(path) as f:
        j = f.read()
    ws = Workspace.model_validate_json(j)
    # Metadata is added after loading. This way code changes take effect on old boxes too.
    _update_metadata(ws)
    return ws


def _update_metadata(ws):
    catalog = ops.CATALOGS.get(ws.env, {})
    nodes = {node.id: node for node in ws.nodes}
    done = set()
    while len(done) < len(nodes):
        for node in ws.nodes:
            if node.id in done:
                continue
            data = node.data
            if node.parentId is None:
                op = catalog.get(data.title)
            elif node.parentId not in nodes:
                data.error = f'Parent not found: {node.parentId}'
                done.add(node.id)
                continue
            elif node.parentId in done:
                op = nodes[node.parentId].data.meta.sub_nodes[data.title]
            else:
                continue
            if op:
                data.meta = op
                node.type = op.type
                if data.error == 'Unknown operation.':
                    data.error = None
            else:
                data.error = 'Unknown operation.'
            done.add(node.id)
    return ws
