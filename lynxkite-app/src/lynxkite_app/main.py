"""The FastAPI server for serving the LynxKite application."""

import os
import shutil
import pydantic
import fastapi
import importlib
import pathlib
import pkgutil
from fastapi.staticfiles import StaticFiles
import starlette
from lynxkite.core import ops
from lynxkite.core import workspace
from . import crdt

if os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").strip().lower() == "true":
    import cudf.pandas

    cudf.pandas.install()


def detect_plugins():
    plugins = {}
    for _, name, _ in pkgutil.iter_modules():
        if name.startswith("lynxkite_"):
            print(f"Importing {name}")
            plugins[name] = importlib.import_module(name)
    if not plugins:
        print("No LynxKite plugins found. Be sure to install some!")
    return plugins


lynxkite_plugins = detect_plugins()

app = fastapi.FastAPI(lifespan=crdt.lifespan)
app.include_router(crdt.router)


@app.get("/api/catalog")
def get_catalog():
    return {
        k: {op.name: op.model_dump() for op in v.values()}
        for k, v in ops.CATALOGS.items()
    }


class SaveRequest(workspace.BaseConfig):
    path: str
    ws: workspace.Workspace


def save(req: SaveRequest):
    path = DATA_PATH / req.path
    assert path.is_relative_to(DATA_PATH)
    workspace.save(req.ws, path)


@app.post("/api/save")
async def save_and_execute(req: SaveRequest):
    save(req)
    await workspace.execute(req.ws)
    save(req)
    return req.ws


@app.post("/api/delete")
async def delete_workspace(req: dict):
    json_path: pathlib.Path = DATA_PATH / req["path"]
    crdt_path: pathlib.Path = CRDT_PATH / f"{req['path']}.crdt"
    assert json_path.is_relative_to(DATA_PATH)
    assert crdt_path.is_relative_to(CRDT_PATH)
    json_path.unlink()
    crdt_path.unlink()


@app.get("/api/load")
def load(path: str):
    path = DATA_PATH / path
    assert path.is_relative_to(DATA_PATH)
    if not path.exists():
        return workspace.Workspace()
    return workspace.load(path)


DATA_PATH = pathlib.Path(os.environ.get("LYNXKITE_DATA", "lynxkite_data"))
CRDT_PATH = pathlib.Path(os.environ.get("LYNXKITE_CRDT_DATA", "lynxkite_crdt_data"))


class DirectoryEntry(pydantic.BaseModel):
    name: str
    type: str


@app.get("/api/dir/list")
def list_dir(path: str):
    path = DATA_PATH / path
    assert path.is_relative_to(DATA_PATH)
    return sorted(
        [
            DirectoryEntry(
                p.relative_to(DATA_PATH), "directory" if p.is_dir() else "workspace"
            )
            for p in path.iterdir()
        ]
    )


@app.post("/api/dir/mkdir")
def make_dir(req: dict):
    path = DATA_PATH / req["path"]
    assert path.is_relative_to(DATA_PATH)
    assert not path.exists()
    path.mkdir()
    return list_dir(path.parent)


@app.post("/api/dir/delete")
def delete_dir(req: dict):
    path: pathlib.Path = DATA_PATH / req["path"]
    assert all([path.is_relative_to(DATA_PATH), path.exists(), path.is_dir()])
    shutil.rmtree(path)
    return list_dir(path.parent)


@app.get("/api/service/{module_path:path}")
async def service_get(req: fastapi.Request, module_path: str):
    """Executors can provide extra HTTP APIs through the /api/service endpoint."""
    module = lynxkite_plugins[module_path.split("/")[0]]
    return await module.api_service_get(req)


@app.post("/api/service/{module_path:path}")
async def service_post(req: fastapi.Request, module_path: str):
    """Executors can provide extra HTTP APIs through the /api/service endpoint."""
    module = lynxkite_plugins[module_path.split("/")[0]]
    return await module.api_service_post(req)


class SPAStaticFiles(StaticFiles):
    """Route everything to index.html. https://stackoverflow.com/a/73552966/3318517"""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (
            fastapi.HTTPException,
            starlette.exceptions.HTTPException,
        ) as ex:
            if ex.status_code == 404:
                return await super().get_response(".", scope)
            else:
                raise ex


static_dir = SPAStaticFiles(packages=[("lynxkite_app", "web_assets")], html=True)
app.mount("/", static_dir, name="web_assets")
