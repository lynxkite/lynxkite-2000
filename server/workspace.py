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

class Workspace(BaseConfig):
    nodes: list[WorkspaceNode] = dataclasses.field(default_factory=list)
    edges: list[WorkspaceEdge] = dataclasses.field(default_factory=list)


def execute(ws):
    # Nodes are responsible for interpreting/executing their child nodes.
    nodes = [n for n in ws.nodes if not n.parentId]
    children = {}
    for n in ws.nodes:
        if n.parentId:
            children.setdefault(n.parentId, []).append(n)
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
                if len(op.inputs) == 1 and op.inputs.get('multi') == '*':
                    # It's a flexible input. Create n+1 handles.
                    data.inputs = {f'input{i}': None for i in range(len(inputs) + 1)}
                data.error = None
                outputs[node.id] = output
                if op.type == 'visualization' or op.type == 'table_view':
                    data.view = output


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
    nodes = {node.id: node for node in ws.nodes}
    done = set()
    while len(done) < len(nodes):
        for node in ws.nodes:
            if node.id in done:
                continue
            data = node.data
            if node.parentId is None:
                op = ops.ALL_OPS.get(data.title)
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
