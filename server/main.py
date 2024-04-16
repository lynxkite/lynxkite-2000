from typing import Optional
import fastapi
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


@app.post("/api/save")
def save(ws: Workspace):
    print(ws)
    execute(ws)
    print('exec done', ws)
    return ws
