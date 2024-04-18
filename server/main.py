from typing import Optional
import dataclasses
import fastapi
import json
import pathlib
import pydantic
import traceback
from . import ops
from . import basic_ops

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

class WorkspaceNode(BaseConfig):
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position

class WorkspaceEdge(BaseConfig):
    id: str
    source: str
    target: str

class Workspace(BaseConfig):
    nodes: list[WorkspaceNode]
    edges: list[WorkspaceEdge]


app = fastapi.FastAPI()


@app.get("/api/catalog")
def get_catalog():
    return [
        {
          'type': op.type,
          'data': { 'title': op.name, 'params': op.params },
          'targetPosition': 'left' if op.inputs else None,
          'sourcePosition': 'right' if op.outputs else None,
        }
        for op in ops.ALL_OPS.values()]

def execute(ws):
    nodes = ws.nodes
    outputs = {}
    failed = 0
    while len(outputs) + failed < len(nodes):
        for node in nodes:
            if node.id in outputs:
                continue
            inputs = [edge.source for edge in ws.edges if edge.target == node.id]
            if all(input in outputs for input in inputs):
                inputs = [outputs[input] for input in inputs]
                data = node.data
                op = ops.ALL_OPS[data.title]
                try:
                  output = op(*inputs, **data.params)
                except Exception as e:
                  traceback.print_exc()
                  data.error = str(e)
                  failed += 1
                  continue
                data.error = None
                outputs[node.id] = output
                if op.type == 'graph_view' or op.type == 'table_view':
                    data.view = output


class SaveRequest(BaseConfig):
    path: str
    ws: Workspace

def save(req: SaveRequest):
    path = DATA_PATH / req.path
    assert path.is_relative_to(DATA_PATH)
    j = req.ws.model_dump_json(indent=2)
    with open(path, 'w') as f:
        f.write(j)

@app.post("/api/save")
def save_and_execute(req: SaveRequest):
    save(req)
    execute(req.ws)
    save(req)
    return req.ws

@app.get("/api/load")
def load(path: str):
    path = DATA_PATH / path
    assert path.is_relative_to(DATA_PATH)
    if not path.exists():
        return Workspace(nodes=[], edges=[])
    with open(path) as f:
        j = f.read()
    ws = Workspace.model_validate_json(j)
    return ws

DATA_PATH = pathlib.Path.cwd() / 'data'

@dataclasses.dataclass(order=True)
class DirectoryEntry:
    name: str
    type: str

@app.get("/api/dir/list")
def list_dir(path: str):
    path = DATA_PATH / path
    assert path.is_relative_to(DATA_PATH)
    return sorted([
        DirectoryEntry(p.relative_to(DATA_PATH), 'directory' if p.is_dir() else 'workspace')
        for p in path.iterdir()])

@app.post("/api/dir/mkdir")
def make_dir(req: dict):
    path = DATA_PATH / req['path']
    assert path.is_relative_to(DATA_PATH)
    assert not path.exists()
    path.mkdir()
    return list_dir(path.parent)
