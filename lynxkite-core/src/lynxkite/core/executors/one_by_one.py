"""A LynxKite executor that assumes most operations operate on their input one by one."""

from .. import ops
from .. import workspace
import orjson
import pandas as pd
import pydantic
import traceback
import inspect
import typing


class Context(ops.BaseConfig):
    """Passed to operation functions as "_ctx" if they have such a parameter."""

    node: workspace.WorkspaceNode
    last_result: typing.Any = None


class Output(ops.BaseConfig):
    """Return this to send values to specific outputs of a node."""

    output_handle: str
    value: dict


def df_to_list(df):
    return df.to_dict(orient="records")


def has_ctx(op):
    sig = inspect.signature(op.func)
    return "_ctx" in sig.parameters


CACHES = {}


def register(env: str, cache: bool = True):
    """Registers the one-by-one executor."""
    if cache:
        CACHES[env] = {}
        cache = CACHES[env]
    else:
        cache = None
    ops.EXECUTORS[env] = lambda ws: execute(ws, ops.CATALOGS[env], cache=cache)


def get_stages(ws: workspace.Workspace, catalog: dict[str, ops.Op]):
    """Inputs on top/bottom are batch inputs. We decompose the graph into a DAG of components along these edges."""
    nodes = {n.id: n for n in ws.nodes}
    batch_inputs = {}
    inputs = {}
    # For each edge in the workspacce, we record the inputs (sources)
    # required for each node (target).
    for edge in ws.edges:
        inputs.setdefault(edge.target, []).append(edge.source)
        node = nodes[edge.target]
        op = catalog[node.data.title]
        i = op.inputs[edge.targetHandle]
        if i.side in [ops.Side.TOP, ops.Side.BOTTOM]:
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


def _default_serializer(obj):
    if isinstance(obj, pydantic.BaseModel):
        return obj.dict()
    return {"__nonserializable__": id(obj)}


def make_cache_key(obj):
    return orjson.dumps(obj, default=_default_serializer)


EXECUTOR_OUTPUT_CACHE = {}


async def await_if_needed(obj):
    if inspect.isawaitable(obj):
        return await obj
    return obj


async def execute(ws: workspace.Workspace, catalog: dict[str, ops.Op], cache=None):
    nodes = {n.id: n for n in ws.nodes}
    contexts = {n.id: Context(node=n) for n in ws.nodes}
    edges = {n.id: [] for n in ws.nodes}
    for e in ws.edges:
        edges[e.source].append(e)
    tasks = {}
    NO_INPUT = object()  # Marker for initial tasks.
    for node in ws.nodes:
        node.data.error = None
        op = catalog.get(node.data.title)
        if op is None:
            node.data.error = f'Operation "{node.data.title}" not found.'
            continue
        # Start tasks for nodes that have no non-batch inputs.
        if all(
            [
                i.side in [ops.Side.TOP, ops.Side.BOTTOM]
                for i in op.inputs.values()
            ]
        ):
            tasks[node.id] = [NO_INPUT]
    batch_inputs = {}
    # Run the rest until we run out of tasks.
    stages = get_stages(ws, catalog)
    for stage in stages:
        next_stage = {}
        while tasks:
            n, ts = tasks.popitem()
            if n not in stage:
                next_stage.setdefault(n, []).extend(ts)
                continue
            node = nodes[n]
            data = node.data
            op = catalog[data.title]
            params = {**data.params}
            if has_ctx(op):
                params["_ctx"] = contexts[node.id]
            results = []
            for task in ts:
                try:
                    inputs = []
                    for i in op.inputs.values():
                        if i.side in [ops.Side.TOP, ops.Side.BOTTOM]:
                            assert (n, i.name) in batch_inputs, f"{i.name} is missing"
                            inputs.append(batch_inputs[(n, i.name)])
                        else:
                            inputs.append(task)
                    if cache is not None:
                        key = make_cache_key((inputs, params))
                        if key not in cache:
                            cache[key] = await await_if_needed(op(*inputs, **params))
                        result = cache[key]
                    else:
                        result = await await_if_needed(op(*inputs, **params))
                except Exception as e:
                    traceback.print_exc()
                    data.error = str(e)
                    break
                contexts[node.id].last_result = result
                # Returned lists and DataFrames are considered multiple tasks.
                if isinstance(result, pd.DataFrame):
                    result = df_to_list(result)
                elif not isinstance(result, list):
                    result = [result]
                results.extend(result)
            else:  # Finished all tasks without errors.
                if (
                    op.view_type == ops.ViewType.VISUALIZATION
                    or op.view_type == ops.ViewType.TABLE_VIEW
                    or op.view_type == ops.ViewType.IMAGE
                ):
                    data.display = results[0]
                for edge in edges[node.id]:
                    t = nodes[edge.target]
                    op = catalog[t.data.title]
                    i = op.inputs[edge.targetHandle]
                    if i.side in [ops.Side.TOP, ops.Side.BOTTOM]:
                        batch_inputs.setdefault(
                            (edge.target, edge.targetHandle), []
                        ).extend(results)
                    else:
                        tasks.setdefault(edge.target, []).extend(results)
        tasks = next_stage
    return contexts
