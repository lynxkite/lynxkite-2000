"""The FastAPI server for serving the LynxKite application."""

import asyncio
import contextlib
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
import typing

mem = joblib.Memory(".joblib-cache", verbose=0)
ops.CACHE_WRAPPER = mem.cache

enable_thread_proxies()
opcontext.TERMINAL_EMULATOR = capture_output
opcontext.PROGRESS_REPORTER = ProgressReporter
opcontext.TQDM_CAPTURER = capture_tqdm
lynxkite_plugins = ops.detect_plugins()
ops.save_catalogs("plugins loaded")


def _compute_nims() -> list[dict]:
    """Build a list of NIM status dicts from live Kubernetes deployments."""
    from kubernetes import client

    _load_k8s_config()
    active_workspaces = _get_active_workspaces()
    nims = []
    for d in client.AppsV1Api().list_deployment_for_all_namespaces().items:
        labels = dict(d.metadata.labels) if d.metadata.labels else {}
        used_by = [w for w in [labels.get("workspace")] if w]
        for ws_name, ws in active_workspaces:
            for node in getattr(ws, "nodes", []):
                node_name = getattr(node.data, "op_id", None) or getattr(node.data, "title", None)
                if node_name and node_name == d.metadata.name and ws_name not in used_by:
                    used_by.append(ws_name)
                    break
        nims.append(
            {
                "publisher": d.metadata.namespace or "",
                "name": d.metadata.name,
                "status": "running" if (d.status.available_replicas or 0) > 0 else "stopped",
                "replicasHealthy": d.status.available_replicas or 0,
                "replicasRequested": d.spec.replicas or 0,
                "usedByWorkspaces": used_by,
            }
        )
    return nims


async def _progress_refresh_loop():
    """Background task: refresh K8s GPU counts and NIM data into the progress CRDT doc."""
    while True:
        # K8s GPU fallback for workspace entries.
        k8s_workspace_gpus: dict[str, int] = {}
        try:
            from kubernetes import client as _k8s_client

            _load_k8s_config()
            for d in _k8s_client.AppsV1Api().list_deployment_for_all_namespaces().items:
                labels = dict(d.metadata.labels) if d.metadata.labels else {}
                ws_label = labels.get("workspace")
                if ws_label:
                    k8s_workspace_gpus[ws_label] = k8s_workspace_gpus.get(ws_label, 0) + (
                        d.spec.replicas or 0
                    )
        except Exception as e:
            print(f"K8s GPU info refresh error: {e}")
        crdt.update_progress_workspaces(k8s_workspace_gpus)
        # NIMs.
        try:
            nims = _compute_nims()
            crdt.update_progress_nims(nims)
        except Exception as e:
            print(f"NIM refresh error: {e}")
        await asyncio.sleep(15)


@contextlib.asynccontextmanager
async def _lifespan(app):
    async with crdt.lifespan(app):
        task = asyncio.create_task(_progress_refresh_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = fastapi.FastAPI(lifespan=_lifespan)
app.include_router(crdt.router)
app.include_router(icons.router)
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
                name=str(p.relative_to(data_path)),
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


def _load_k8s_config():
    """Load Kubernetes configuration (in-cluster or from kubeconfig)."""
    from kubernetes import config

    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()


def _get_active_workspaces() -> typing.List[typing.Tuple[str, workspace.Workspace]]:
    """Return list of (name, workspace) tuples for all active workspaces."""
    active_workspaces = []
    server = getattr(crdt, "ws_websocket_server", None)
    if server:
        for name, room in getattr(server, "rooms", {}).items():
            try:
                ws = workspace.Workspace.model_validate(room.ws.to_py())
                active_workspaces.append((name, ws))
            except Exception as e:
                print(f"Error loading workspace {name}: {e}")
    return active_workspaces


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
