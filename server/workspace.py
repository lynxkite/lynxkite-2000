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

class WorkspaceNode(BaseConfig):
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position
    parentNode: Optional[str] = None

class WorkspaceEdge(BaseConfig):
    id: str
    source: str
    target: str

class Workspace(BaseConfig):
    nodes: list[WorkspaceNode] = dataclasses.field(default_factory=list)
    edges: list[WorkspaceEdge] = dataclasses.field(default_factory=list)


def execute(ws):
    # Nodes are responsible for interpreting/executing their child nodes.
    nodes = [n for n in ws.nodes if not n.parentNode]
    print(nodes)
    children = {}
    for n in ws.nodes:
        if n.parentNode:
            children.setdefault(n.parentNode, []).append(n)
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
                params = {**data.params}
                if op.sub_nodes:
                    sub_nodes = children.get(node.id, [])
                    sub_node_ids = [node.id for node in sub_nodes]
                    sub_edges = [edge for edge in ws.edges if edge.source in sub_node_ids]
                    params['sub_flow'] = {'nodes': sub_nodes, 'edges': sub_edges}
                try:
                  output = op(*inputs, **params)
                except Exception as e:
                  traceback.print_exc()
                  data.error = str(e)
                  failed += 1
                  continue
                data.error = None
                outputs[node.id] = output
                if op.type == 'graph_view' or op.type == 'table_view':
                    data.view = output


def save(ws: Workspace, path: str):
    j = ws.model_dump_json(indent=2)
    with tempfile.NamedTemporaryFile('w', delete_on_close=False) as f:
        f.write(j)
        f.close()
        os.replace(f.name, path)


def load(path: str):
    with open(path) as f:
        j = f.read()
    ws = Workspace.model_validate_json(j)
    return ws
