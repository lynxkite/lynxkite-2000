"""The FastAPI server for serving the LynxKite application."""

import shutil
import pydantic
import fastapi
import joblib
import pathlib
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import starlette.exceptions
from lynxkite_core import ops
from lynxkite_core import opcontext
from lynxkite_core import workspace
from . import crdt
from . import icons
from .terminal_emulator import capture_output, enable_thread_proxies
from .tqdm_emulator import capture_tqdm, ProgressReporter

try:
    import lynxkite_assistant

    assistant_router: fastapi.APIRouter | None = lynxkite_assistant.router
except ImportError:
    assistant_router = None

try:
    import lynxkite_enterprise.backend as enterprise_backend  # ty: ignore[unresolved-import]
except ImportError:
    enterprise_backend = None

mem = joblib.Memory(".joblib-cache", verbose=0)
ops.CACHE_WRAPPER = mem.cache

enable_thread_proxies()
opcontext.TERMINAL_EMULATOR = capture_output
opcontext.PROGRESS_REPORTER = ProgressReporter
opcontext.TQDM_CAPTURER = capture_tqdm
lynxkite_plugins = ops.detect_plugins()
ops.save_catalogs("plugins loaded")

app = fastapi.FastAPI(lifespan=crdt.lifespan)
app.include_router(crdt.router)
app.include_router(icons.router)
if assistant_router is not None:
    app.include_router(assistant_router)
if enterprise_backend is not None:
    enterprise_backend.register_routes(app, crdt)
app.add_middleware(GZipMiddleware)  # ty: ignore[invalid-argument-type]


def _get_ops(env: str):
    catalog = ops.CATALOGS[env]
    res = {op.name: op.model_dump() for op in catalog.values()}
    res.setdefault("Comment", ops.COMMENT_OP.model_dump())
    return res


@app.get("/api/catalog")
def get_catalog(workspace: str):
    ops.load_user_scripts(workspace)
    return {env: _get_ops(env) for env in ops.CATALOGS}


@app.get("/api/config")
def get_config() -> dict[str, bool]:
    return {
        "assistant_available": assistant_router is not None,
        "enterprise_available": enterprise_backend is not None,
    }


data_path = pathlib.Path()


@app.post("/api/delete")
async def delete_workspace(req: dict):
    json_path: pathlib.Path = data_path / req["path"]
    crdt_path: pathlib.Path = data_path / ".crdt" / f"{req['path']}.crdt"
    assert json_path.is_relative_to(data_path), f"Path '{json_path}' is invalid"
    json_path.unlink()
    crdt_path.unlink()
    crdt.delete_room(req["path"])


class DirectoryEntry(pydantic.BaseModel):
    name: str
    type: str


def _get_path_type(path: pathlib.Path) -> str:
    if path.is_dir():
        return "directory"
    elif path.suffixes[-2:] == [".lynxkite", ".json"]:
        return "workspace"
    else:
        return "file"


@app.get("/api/dir/list")
def list_dir(path: str):
    dir_path = data_path / path
    assert dir_path.is_relative_to(data_path), f"Path '{dir_path}' is invalid"
    return sorted(
        [
            DirectoryEntry(
                name=p.relative_to(data_path).as_posix(),
                type=_get_path_type(p),
            )
            for p in dir_path.iterdir()
            if not p.name.startswith(".")
        ],
        key=lambda x: (x.type != "directory", x.name.lower()),
    )


@app.post("/api/dir/mkdir")
def make_dir(req: dict):
    path = data_path / req["path"]
    assert path.is_relative_to(data_path), f"Path '{path}' is invalid"
    assert not path.exists(), f"{path} already exists"
    path.mkdir()


@app.post("/api/dir/delete")
def delete_dir(req: dict):
    path: pathlib.Path = data_path / req["path"]
    assert all([path.is_relative_to(data_path), path.exists(), path.is_dir()]), (
        f"Path '{path}' is invalid"
    )
    shutil.rmtree(path)


@app.post("/api/rename")
def rename_path(req: dict):
    old_path: pathlib.Path = data_path / req["old_path"]
    new_path: pathlib.Path = data_path / req["new_path"]
    assert old_path.is_relative_to(data_path), f"Path '{old_path}' is invalid"
    assert new_path.is_relative_to(data_path), f"Path '{new_path}' is invalid"
    assert old_path.exists(), f"Path '{old_path}' does not exist"
    assert not new_path.exists(), f"Path '{new_path}' already exists"
    old_rel = req["old_path"]
    old_path.rename(new_path)
    # Drop any open room under the old name so clients don't keep stale pointers.
    crdt.delete_room(old_rel)


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


@app.post("/api/upload")
async def upload(req: fastapi.Request):
    """Receives file uploads and stores them in DATA_PATH."""
    form = await req.form()
    for file in form.values():
        if not isinstance(file, fastapi.UploadFile) or not file.filename:
            continue
        file_path = data_path / "uploads" / file.filename
        assert file_path.is_relative_to(data_path), f"Path '{file_path}' is invalid"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"status": "ok"}


@app.post("/api/execute_workspace")
async def execute_workspace(name: str):
    """Trigger and await the execution of a workspace."""
    room = await crdt.get_room(name)
    ws_pyd = workspace.Workspace.model_validate(room.ws.to_py())
    await crdt.execute(name, room.ws, ws_pyd)


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
