"""
A LynxKite executor that assumes most operations operate on their input one by one.
"""

import contextlib
import urllib.parse
from .. import ops
from .. import workspace
import traceback
import inspect
import typing


class Context(ops.BaseConfig):
    """Passed to operation functions as "_ctx" if they have such a parameter.

    Attributes:
        node: The workspace node that this context is associated with.
        last_result: The last result produced by the operation.
            This can be used to incrementally build a result, when the operation
            is executed for multiple items.
    """

    node: workspace.WorkspaceNode
    last_result: typing.Any = None


def _df_to_list(df):
    return df.to_dict(orient="records")


def _has_ctx(op):
    sig = inspect.signature(op.func)
    return "_ctx" in sig.parameters


def register(env: str, cache: bool = True):
    """Registers the one-by-one executor.

    Usage:

        from lynxkite_core.executors import one_by_one
        one_by_one.register("My Environment")
    """
    ops.EXECUTORS[env] = lambda ws, ctx: _execute(ws, ops.CATALOGS[env], ctx)


def _get_stages(ws, catalog: ops.Catalog):
    """Inputs on top/bottom are batch inputs. We decompose the graph into a DAG of components along these edges."""
    nodes = {n.id: n for n in ws.nodes}
    batch_inputs = {}
    inputs = {}
    # For each edge in the workspacce, we record the inputs (sources)
    # required for each node (target).
    for edge in ws.edges:
        inputs.setdefault(edge.target, []).append(edge.source)
        node = nodes[edge.target]
        op = catalog[node.data.op_id]
        if op.get_input(edge.targetHandle).position.is_vertical():
            batch_inputs.setdefault(edge.target, []).append(edge.source)
    stages = []
    for bt, bss in batch_inputs.items():
        upstream = set(bss)
        new = set(bss)
        while new:
            n = new.pop()
            for i in inputs.get(n, []):
                if i not in upstream:
                    upstream.add(i)
                    new.add(i)
        stages.append(upstream)
    stages.sort(key=lambda s: len(s))
    stages.append(set(nodes))
    return stages


async def _await_if_needed(obj):
    if inspect.isawaitable(obj):
        return await obj
    return obj


async def _execute(
    ws: workspace.Workspace,
    catalog: ops.Catalog,
    ctx: workspace.WorkspaceExecutionContext | None = None,
):
    nodes = {n.id: n for n in ws.nodes}
    contexts = {n.id: Context(node=n) for n in ws.nodes}
    edges = {n.id: [] for n in ws.nodes}
    for e in ws.edges:
        edges[e.source].append(e)
    tasks = {}
    NO_INPUT = object()  # Marker for initial tasks.
    for node in ws.nodes:
        op = catalog.get(node.data.op_id)
        if op is None:
            node.publish_error(f'Operation "{node.data.op_id}" not found.')
            continue
        node.publish_error(None)
        # Start tasks for nodes that have no non-batch inputs.
        if all([i.position.is_vertical() for i in op.inputs]):
            tasks[node.id] = [NO_INPUT]
    batch_inputs = {}
    # Run the rest until we run out of tasks.
    stages = _get_stages(ws, catalog)
    for stage in stages:
        next_stage = {}
        while tasks:
            n, ts = tasks.popitem()
            if n not in stage:
                next_stage.setdefault(n, []).extend(ts)
                continue
            node = nodes[n]
            op = catalog[node.data.op_id]
            params = {**node.data.params}
            if _has_ctx(op):
                params["_ctx"] = contexts[node.id]
            results = []
            node.publish_started()
            for task in ts:
                try:
                    inputs = []
                    missing = []
                    for i in op.inputs:
                        if i.position.is_vertical():
                            if (n, i.name) in batch_inputs:
                                inputs.append(batch_inputs[(n, i.name)])
                            else:
                                opt_type = ops.get_optional_type(i.type)
                                if opt_type is not None:
                                    inputs.append(None)
                                else:
                                    missing.append(i.name)
                        else:
                            inputs.append(task)
                    if missing:
                        node.publish_error(f"Missing input: {', '.join(missing)}")
                        break
                    result = op(*inputs, **params)
                    output = await _await_if_needed(result.output)
                except Exception as e:
                    traceback.print_exc()
                    node.publish_error(e)
                    break
                contexts[node.id].last_result = output
                # Returned lists and DataFrames are considered multiple tasks.
                if hasattr(output, "to_dict"):
                    output = _df_to_list(output)
                elif not isinstance(output, list):
                    output = [output]
                results.extend(output)
            else:  # Finished all tasks without errors.
                if op.type == "gradio" and ctx and ctx.app:
                    url = f"/api/lynxkite_graph_analytics/{ws.path}/{node.id}"
                    await mount_gradio(ctx.app, result.output, url)
                    result.display = {"backend": urllib.parse.quote(url)}
                    result.output = None
                if result.display:
                    result.display = await _await_if_needed(result.display)
                for edge in edges[node.id]:
                    t = nodes[edge.target]
                    op = catalog[t.data.op_id]
                    if op.get_input(edge.targetHandle).position.is_vertical():
                        batch_inputs.setdefault((edge.target, edge.targetHandle), []).extend(
                            results
                        )
                    else:
                        tasks.setdefault(edge.target, []).extend(results)
                node.publish_result(result)
        tasks = next_stage
    return contexts


class _ProxyApp:
    def __init__(self, app):
        self._app = app
        self.router = self

    @contextlib.asynccontextmanager
    async def lifespan_context(self, app):
        yield

    def mount(self, path, gradio_app):
        import starlette.routing  # ty: ignore[unresolved-import]

        router = self._app.router
        route = starlette.routing.Mount(path, gradio_app)
        # Overwrite existing route if it exists.
        for i, r in enumerate(router.routes):
            if r.path == path:
                router.routes[i] = route
                break
        else:
            router.routes.insert(0, route)


async def mount_gradio(app, gradio_app, path: str):
    """Mounts a Gradio app onto a Starlette/FastAPI app at the given path."""
    import gradio as gr  # ty: ignore[unresolved-import]

    app = _ProxyApp(app)
    gr.mount_gradio_app(app, gradio_app, path=path)
    # Trigger Gradio lifetime hooks.
    async with app.lifespan_context(app):
        pass
