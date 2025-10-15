"""A LynxKite executor that simply passes the output of one box to the other."""

import os
from .. import ops
from .. import workspace
import traceback
import inspect
import graphlib


def register(env: str):
    """Registers the simple executor.

    Usage:

        from lynxkite_core.executors import simple
        simple.register("My Environment")
    """
    ops.EXECUTORS[env] = lambda ws, _ctx: execute(ws, ops.CATALOGS[env])


async def await_if_needed(obj):
    if inspect.isawaitable(obj):
        return await obj
    return obj


async def execute(ws: workspace.Workspace, catalog: ops.Catalog):
    nodes = {n.id: n for n in ws.nodes}
    dependencies = {n: [] for n in nodes}
    in_edges = {n: {} for n in nodes}
    for e in ws.edges:
        dependencies[e.target].append(e.source)
        assert e.targetHandle not in in_edges[e.target], f"Duplicate input for {e.target}"
        in_edges[e.target][e.targetHandle] = e.source, e.sourceHandle
    outputs = {}
    ts = graphlib.TopologicalSorter(dependencies)
    for node_id in ts.static_order():
        node = nodes[node_id]
        op = catalog[node.data.op_id]
        params = {**node.data.params}
        node.publish_started()
        try:
            inputs = []
            missing = []
            for i in op.inputs:
                edges = in_edges[node_id]
                if i.name in edges and edges[i.name] in outputs:
                    inputs.append(outputs[edges[i.name]])
                else:
                    opt_type = ops.get_optional_type(i.type)
                    if opt_type is not None:
                        inputs.append(None)
                    else:
                        missing.append(i.name)
            if missing:
                node.publish_error(f"Missing input: {', '.join(missing)}")
                continue
            result = op(*inputs, **params)
            result.output = await await_if_needed(result.output)
            result.display = await await_if_needed(result.display)
            if len(op.outputs) == 1:
                [output] = op.outputs
                outputs[node_id, output.name] = result.output
            elif len(op.outputs) > 1:
                assert type(result.output) is dict, "An op with multiple outputs must return a dict"
                for output in op.outputs:
                    outputs[node_id, output.name] = result.output[output.name]
            node.publish_result(result)
        except Exception as e:
            if not os.environ.get("LYNXKITE_SUPPRESS_OP_ERRORS"):
                traceback.print_exc()
            node.publish_error(e)
    return outputs
