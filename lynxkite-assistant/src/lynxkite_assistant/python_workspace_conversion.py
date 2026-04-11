"""
Convert a series of Python function calls into a workspace, or the other way.
"""

import ast
import graphlib
from lynxkite_core import workspace


def python_to_workspace(code: str) -> workspace.Workspace:
    tree = ast.parse(code)
    ws = workspace.Workspace()
    saved_values = {}
    box_x = {}
    box_y = {}
    for s in tree.body:
        src = ast.get_source_segment(code, s)
        error_msg = f"Unexpected statement on line {s.lineno}:\n\n  {src}\n\nThe file must only contain function calls. Keyword arguments must be constants or previous results. Positional arguments are not allowed."
        save_as = None
        if isinstance(s, ast.Assign):
            assert len(s.targets) == 1, error_msg
            assert isinstance(s.targets[0], ast.Name), error_msg
            save_as = s.targets[0].id
        assert isinstance(s, ast.Assign | ast.Expr), error_msg
        s = s.value
        assert isinstance(s, ast.Call), error_msg
        assert isinstance(s.func, ast.Name), error_msg
        box_id = f"{s.func.id} on line {s.lineno}"
        assert len(s.args) == 0, error_msg
        kwargs = {}
        for kw in s.keywords:
            if kw.arg:
                kwargs[kw.arg] = kw.value
        params = {}
        x = 0
        lowest_input = None
        for arg_name, arg_value in kwargs.items():
            assert isinstance(arg_value, ast.Constant | ast.Name), error_msg
            if isinstance(arg_value, ast.Constant):
                params[arg_name] = arg_value.value
            elif isinstance(arg_value, ast.Name):
                src = saved_values[arg_value.id]
                x = max(x, box_x[src] + 1)
                if lowest_input is None or box_y[src] < lowest_input:
                    lowest_input = box_y[src]
                ws.add_edge(src, "output", box_id, arg_name)
        taken = {box_y[b] for b in box_y if box_x[b] == x}
        y = lowest_input or 0
        while y in taken:
            y += 1
        box_x[box_id] = x
        box_y[box_id] = y
        ws.add_node(
            id=box_id,
            title=s.func.id,
            op_id=box_id,
            params=params,
            width=400,
            height=400,
            position=workspace.Position(x=x * 500, y=y * 450),
        )
        if save_as:
            saved_values[save_as] = box_id
    ws.update_metadata()
    return ws


def workspace_to_python(ws: workspace.Workspace) -> str:
    code = [
        '"""The Python representation of the workspace."""',
        "# Imports are handled automatically.",
    ]
    node_by_id = {node.id: node for node in ws.nodes}
    incoming_edges: dict[str, list[workspace.WorkspaceEdge]] = {node.id: [] for node in ws.nodes}
    outgoing_count: dict[str, int] = {node.id: 0 for node in ws.nodes}
    dependencies: dict[str, set[str]] = {node.id: set() for node in ws.nodes}
    for edge in ws.edges:
        # Ignore broken edges that point to missing nodes.
        if edge.source not in node_by_id or edge.target not in node_by_id:
            continue
        incoming_edges[edge.target].append(edge)
        outgoing_count[edge.source] += 1
        dependencies[edge.target].add(edge.source)

    sorter = graphlib.TopologicalSorter(dependencies)
    sorted_node_ids = list(sorter.static_order())
    saved_values: dict[str, str] = {}
    variable_names: dict[str, str] = {}
    next_var_index = 1

    for node_id in sorted_node_ids:
        node = node_by_id[node_id]
        kwargs = dict(getattr(node.data, "params", {}))
        for edge in sorted(incoming_edges[node_id], key=lambda e: e.targetHandle):
            if edge.source in variable_names:
                kwargs[edge.targetHandle] = saved_values[edge.source]

        kwarg_parts = [f"{name}={repr(value)}" for name, value in kwargs.items()]
        call = f"{node.data.title}({', '.join(kwarg_parts)})"
        if outgoing_count[node_id] > 0:
            variable_name = f"v{next_var_index}"
            next_var_index += 1
            variable_names[node_id] = variable_name
            saved_values[node_id] = variable_name
            code.append(f"{variable_name} = {call}")
        else:
            code.append(call)
    return "\n".join(code)
