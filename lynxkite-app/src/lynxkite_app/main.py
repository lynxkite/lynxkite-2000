"""The FastAPI server for serving the LynxKite application."""

import asyncio
import contextlib
import importlib
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

_k8s_unavailable_logged = False


def _log_k8s_unavailable_once() -> None:
    global _k8s_unavailable_logged
    if _k8s_unavailable_logged:
        return
    print("Kubernetes package is not installed; NIM/K8s progress features are disabled.")
    _k8s_unavailable_logged = True


def _k8s_client_module():
    try:
        return importlib.import_module("kubernetes.client")
    except ModuleNotFoundError:
        _log_k8s_unavailable_once()
        return None


def _k8s_config_module():
    try:
        return importlib.import_module("kubernetes.config")
    except ModuleNotFoundError:
        _log_k8s_unavailable_once()
        return None


def _compute_nims() -> list[dict]:
    """Build a list of NIM status dicts from live Kubernetes deployments."""
    client = _k8s_client_module()
    if client is None:
        return []

    try:
        if not _load_k8s_config():
            return []
        system_namespaces = {"kube-system", "kube-public", "kube-node-lease"}
        active_workspaces = _get_active_workspaces()
        nims = []
        for d in client.AppsV1Api().list_deployment_for_all_namespaces().items:
            if (d.metadata.namespace or "") in system_namespaces:
                continue
            labels = dict(d.metadata.labels) if d.metadata.labels else {}
            used_by = [w for w in [labels.get("workspace")] if w]
            for ws_name, ws in active_workspaces:
                for node in getattr(ws, "nodes", []):
                    node_name = getattr(node.data, "op_id", None) or getattr(
                        node.data, "title", None
                    )
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
    except Exception as e:
        print(f"NIM refresh error: {e}")
        return []


def _get_k8s_workspace_gpus() -> dict[str, int]:
    """Get workspace GPU counts from Kubernetes deployment labels."""
    k8s_workspace_gpus: dict[str, int] = {}
    client = _k8s_client_module()
    if client is None:
        return k8s_workspace_gpus
    try:
        if not _load_k8s_config():
            return k8s_workspace_gpus
        for d in client.AppsV1Api().list_deployment_for_all_namespaces().items:
            labels = dict(d.metadata.labels) if d.metadata.labels else {}
            ws_label = labels.get("workspace")
            if ws_label:
                k8s_workspace_gpus[ws_label] = k8s_workspace_gpus.get(ws_label, 0) + (
                    d.spec.replicas or 0
                )
    except Exception as e:
        print(f"K8s GPU info refresh error: {e}")
    return k8s_workspace_gpus


async def _progress_refresh_loop():
    """Background task: refresh K8s GPU counts and NIM data into the progress CRDT doc."""
    while True:
        crdt.update_progress_workspaces(_get_k8s_workspace_gpus())
        crdt.update_progress_nims(_compute_nims())
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


@app.post("/api/pause_workspace")
async def pause_workspace(req: dict):
    """Pause or resume a workspace."""
    room_name = req.get("room_name")
    if not isinstance(room_name, str) or not room_name:
        raise fastapi.HTTPException(status_code=400, detail="Missing or invalid room_name")
    room = await crdt.get_room(room_name)
    paused = req.get("paused", True)
    with room.ws.doc.transaction():
        room.ws["paused"] = paused
    return {"status": "ok", "room_name": room_name, "paused": paused}


@app.post("/api/stop_workspace")
async def stop_workspace(req: dict):
    """Stop and reset all nodes in a workspace."""
    room_name = req.get("room_name")
    if not isinstance(room_name, str) or not room_name:
        raise fastapi.HTTPException(status_code=400, detail="Missing or invalid room_name")
    room = await crdt.get_room(room_name)
    with room.ws.doc.transaction():
        room.ws["paused"] = True
        for node in room.ws["nodes"]:
            node["data"]["status"] = "planned"
            node["data"]["message"] = None
    return {"status": "ok", "room_name": room_name}


@app.post("/api/scale_nim")
async def scale_nim(req: dict):
    """Scale a NIM deployment to the specified replica count."""
    client = _k8s_client_module()
    if client is None:
        raise fastapi.HTTPException(
            status_code=503,
            detail="Kubernetes integration is unavailable: python package 'kubernetes' is not installed.",
        )

    name = req.get("name")
    namespace = req.get("namespace") or req.get("publisher")
    replicas = req.get("replicas")

    print(f"Scaling NIM {namespace}/{name} to {replicas} replicas")
    if not isinstance(name, str) or not name:
        raise fastapi.HTTPException(status_code=400, detail="Missing or invalid name")
    if not isinstance(namespace, str) or not namespace:
        raise fastapi.HTTPException(status_code=400, detail="Missing or invalid namespace")
    if not isinstance(replicas, int) or replicas < 0:
        raise fastapi.HTTPException(
            status_code=400, detail="Replicas must be a non-negative integer"
        )

    if not _load_k8s_config():
        raise fastapi.HTTPException(
            status_code=503,
            detail="Kubernetes integration is unavailable: configuration could not be loaded.",
        )
    apps = client.AppsV1Api()
    apps.patch_namespaced_deployment_scale(
        name=name,
        namespace=namespace,
        body={"spec": {"replicas": replicas}},
    )

    # Push a fresh snapshot right away so the UI updates without waiting for poll loop
    crdt.update_progress_workspaces(_get_k8s_workspace_gpus())
    try:
        nims = _compute_nims()
        crdt.update_progress_nims(nims)
    except Exception as e:
        print(f"Error computing/updating NIMs after scaling {namespace}/{name}: {e}")

    return {"status": "ok", "name": name, "namespace": namespace, "replicas": replicas}


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
    config = _k8s_config_module()
    if config is None:
        return False

    try:
        config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            return False
    return True


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
