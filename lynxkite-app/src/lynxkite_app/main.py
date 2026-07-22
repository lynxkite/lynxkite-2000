"""The FastAPI server for serving the LynxKite application."""

import os
import shutil
import pydantic
from pydantic_core import from_json
import fastapi
import joblib
import pathlib
import starlette.datastructures
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import starlette.exceptions
from lynxkite_core import ops
from lynxkite_core import opcontext
from lynxkite_core import workspace
from . import acl
from . import auth
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

try:
    from lynxkite_enterprise.lim_worker import register_lim_routes  # ty: ignore[unresolved-import]
except ImportError:
    register_lim_routes = None

LIM_WORKER = os.environ.get("AM_I_A_LIM_WORKER")

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
if register_lim_routes is not None and LIM_WORKER:
    register_lim_routes(app)
app.add_middleware(GZipMiddleware)  # ty: ignore[invalid-argument-type]


def _get_ops(env: str):
    catalog = ops.CATALOGS[env]
    res = {op.name: op for op in catalog.values()}
    res.setdefault("Comment", ops.COMMENT_OP)
    return res


@app.get("/api/catalog")
async def get_catalog(workspace: str, request: fastapi.Request) -> dict[str, dict[str, ops.Op]]:
    await auth.check_permission(request, "read", workspace)
    ops.load_user_scripts(workspace)
    return {env: _get_ops(env) for env in ops.CATALOGS}


@app.get("/api/config")
def get_config() -> dict[str, bool | str | None]:
    return {
        "assistant_available": assistant_router is not None,
        "authentication_issuer": auth.issuer,
        "authentication_audience": auth.audience,
        "enterprise_available": enterprise_backend is not None,
    }


data_path = pathlib.Path()
acl.set_data_root(data_path)


@app.get("/api/permissions")
async def get_permissions(path: str, request: fastapi.Request) -> dict[str, bool]:
    user = await auth.get_current_user(request)
    return acl.effective_permissions(user, path, auth_enabled=auth.is_auth_enabled())


@app.get("/api/permissions/me")
async def get_permissions_me(request: fastapi.Request) -> dict[str, bool]:
    user = await auth.get_current_user(request)
    return acl.effective_permissions(user, "", auth_enabled=auth.is_auth_enabled())


@app.post("/api/delete")
async def delete_workspace(req: dict, request: fastapi.Request):
    await auth.check_permission(request, "write", req["path"])
    assert isinstance(req["path"], str)
    json_path: pathlib.Path = data_path / req["path"]
    crdt_path: pathlib.Path = data_path / ".crdt" / f"{req['path']}.crdt"
    workspace_files_path = ops.build_output_path(req["path"], "node -1").parent
    assert json_path.is_relative_to(data_path), f"Path '{json_path}' is invalid"
    json_path.unlink()
    crdt_path.unlink()
    if workspace_files_path.exists():
        shutil.rmtree(workspace_files_path)
    crdt.delete_room(req["path"])


@app.get("/api/node_output")
async def get_node_output(workspace: str, node_id: str, version: int, request: fastapi.Request):
    await auth.check_permission(request, "read", f"{workspace}.lynxkite.json")
    json_path = data_path / ops.build_output_path(workspace, node_id)
    assert json_path.is_relative_to(data_path), f"Path '{json_path}' is invalid"
    output = None
    if json_path.exists():
        with open(json_path, mode="r") as f:
            output = from_json(f.read())
    if output is None:
        raise fastapi.HTTPException(status_code=404, detail="Output not found")
    return output


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
async def list_dir(path: str, request: fastapi.Request):
    await auth.check_permission(request, "read", path)
    user = await auth.get_current_user(request)
    dir_path = data_path / path
    assert dir_path.is_relative_to(data_path), f"Path '{dir_path}' is invalid"
    auth_on = auth.is_auth_enabled()
    entries: list[DirectoryEntry] = []
    for p in dir_path.iterdir():
        if p.name.startswith("."):
            continue
        rel = p.relative_to(data_path).as_posix()
        if not acl.has_permission(user, "read", rel, auth_enabled=auth_on):
            continue
        entries.append(DirectoryEntry(name=rel, type=_get_path_type(p)))
    return sorted(entries, key=lambda x: (x.type != "directory", x.name.lower()))


@app.post("/api/dir/mkdir")
async def make_dir(req: dict, request: fastapi.Request):
    await auth.check_permission(request, "write", req["path"])
    path = data_path / req["path"]
    assert path.is_relative_to(data_path), f"Path '{path}' is invalid"
    assert not path.exists(), f"{path} already exists"
    path.mkdir()


@app.post("/api/dir/delete")
async def delete_dir(req: dict, request: fastapi.Request):
    await auth.check_permission(request, "write", req["path"])
    path: pathlib.Path = data_path / req["path"]
    assert all([path.is_relative_to(data_path), path.exists(), path.is_dir()]), (
        f"Path '{path}' is invalid"
    )
    shutil.rmtree(path)


@app.post("/api/rename")
async def rename_path(req: dict, request: fastapi.Request):
    await auth.check_permission(request, "write", req["old_path"])
    await auth.check_permission(request, "write", req["new_path"])
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
    await auth.check_permission(req, "read", "")
    module = lynxkite_plugins[module_path.split("/")[0]]
    return await module.api_service_get(req)


@app.post("/api/service/{module_path:path}")
async def service_post(req: fastapi.Request, module_path: str):
    """Executors can provide extra HTTP APIs through the /api/service endpoint."""
    await auth.check_permission(req, "write", "")
    module = lynxkite_plugins[module_path.split("/")[0]]
    return await module.api_service_post(req)


@app.post("/api/upload")
async def upload(req: fastapi.Request, dir: str = "uploads") -> dict[str, str]:
    """Receives file uploads and stores them in DATA_PATH/dir."""
    await auth.check_permission(req, "write", dir)
    upload_dir = data_path / dir
    assert upload_dir.is_relative_to(data_path), f"Path '{upload_dir}' is invalid"
    form = await req.form()
    for file in form.values():
        if not isinstance(file, starlette.datastructures.UploadFile) or not file.filename:
            continue
        file_path = upload_dir / file.filename
        assert file_path.is_relative_to(data_path), f"Path '{file_path}' is invalid"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"status": "ok"}


@app.post("/api/download")
async def download(req: dict, request: fastapi.Request):
    """Sends a file from DATA_PATH to the client."""
    await auth.check_permission(request, "read", req["path"])
    file_path = data_path / req["path"]
    assert file_path.is_relative_to(data_path), f"Path '{file_path}' is invalid"
    if not file_path.exists() or not file_path.is_file():
        raise fastapi.HTTPException(status_code=404, detail="File not found")
    return fastapi.responses.FileResponse(file_path)


@app.post("/api/execute_workspace")
async def execute_workspace(name: str, req: fastapi.Request):
    """Trigger and await the execution of a workspace."""
    await auth.check_permission(req, "write", name)
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
