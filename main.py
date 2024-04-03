from typing import Union
import fastapi
import pydantic

class Position(pydantic.BaseModel):
    x: float
    y: float

class WorkspaceNode(pydantic.BaseModel):
    id: str
    title: str
    type: str
    position: Position

class WorkspaceConnection(pydantic.BaseModel):
    id: str
    # Baklava.js calls it "from", but that's a reserved keyword in Python.
    src: str = pydantic.Field(None, alias='from')
    dst: str = pydantic.Field(None, alias='to')

class WorkspaceGraph(pydantic.BaseModel):
    nodes: list[WorkspaceNode]
    connections: list[WorkspaceConnection]
    panning: Position
    scaling: float
    nodes: list[WorkspaceNode]

class Workspace(pydantic.BaseModel):
    graph: WorkspaceGraph


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
    return {"status": "ok"}
