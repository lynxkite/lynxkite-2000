from typing import Union
import fastapi
import pydantic
import networkx as nx

class Position(pydantic.BaseModel):
    x: float
    y: float

class WorkspaceNodeData(pydantic.BaseModel):
    title: str
    params: dict

class WorkspaceNode(pydantic.BaseModel):
    id: str
    type: str
    data: WorkspaceNodeData
    position: Position

class WorkspaceEdge(pydantic.BaseModel):
    id: str
    source: str
    target: str

class Workspace(pydantic.BaseModel):
    nodes: list[WorkspaceNode]
    edges: list[WorkspaceEdge]


app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/api/save")
def save(ws: Workspace):
    print(ws)
    G = nx.scale_free_graph(4)
    return {"graph": list(nx.to_edgelist(G))}
