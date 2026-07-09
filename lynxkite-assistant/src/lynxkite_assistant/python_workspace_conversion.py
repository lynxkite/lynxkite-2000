"""
Convert a series of Python function calls into a workspace, or the other way.
"""

import ast
import graphlib
from functools import reduce
from itertools import groupby
from lynxkite_core import ops, workspace
from . import workspace_comments
from .instructions import WORKSPACE_PROMPT


def _get_func_name(func: ast.expr, error_msg: str) -> str:
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        return _get_func_name(func.value, error_msg) + "." + func.attr
    raise AssertionError(error_msg)


def _get_catalog():
    catalog = {}
    for cat in ops.CATALOGS.values():
        for op in cat.values():
            if op.python_function_name:
                catalog[op.python_function_name] = op
    return catalog


def add_edge(arg_name, arg_value_list, saved_values, ws, box_id, error_msg):
    for arg_value in arg_value_list:
        if isinstance(arg_value, ast.Subscript):
            assert isinstance(arg_value.value, ast.Name), error_msg
            assert isinstance(arg_value.slice, (ast.Constant)), error_msg
            name = arg_value.value.id
            sourceHandle = arg_value.slice.value
        else:
            name = arg_value.id
            sourceHandle = "output"
        assert name in saved_values, (
            f"{error_msg}\n\nUnknown variable reference: {name}"
        )
        src = saved_values[name]
        ws.add_edge(src, sourceHandle, box_id, arg_name)


def parse_args(
    box_id,
    kwargs,
    saved_values,
    comment_by_cleaned_text,
    ws,
    error_msg,
    func_name,
    groups,
):
    params = {}
    for arg_name, arg_value in kwargs.items():
        assert isinstance(
            arg_value,
            (ast.Constant, ast.Name, ast.Dict, ast.List, ast.Tuple, ast.Subscript),
        ), error_msg
        if isinstance(arg_value, ast.Constant):
            params[arg_name] = arg_value.value
        elif isinstance(arg_value, (ast.Name, ast.Subscript)):
            add_edge(arg_name, [arg_value], saved_values, ws, box_id, error_msg)
        elif isinstance(arg_value, ast.Dict):
            dict_value = {}
            for key_node, value_node in zip(arg_value.keys, arg_value.values):
                assert isinstance(key_node, ast.Constant) and isinstance(
                    key_node.value, str
                ), error_msg
                assert isinstance(value_node, ast.Constant), error_msg
                dict_value[key_node.value] = value_node.value
            params[arg_name] = dict_value
        elif isinstance(arg_value, ast.List):
            if func_name == "lynxkite_core.ops.group" and all(
                isinstance(item, (ast.Name, ast.Constant)) for item in arg_value.elts
            ):
                boxes = {
                    saved_values[item.id]
                    for item in arg_value.elts
                    if isinstance(item, ast.Name)
                }
                comments = {
                    comment_by_cleaned_text[
                        item.value.replace(" ", "").replace("\n", "")
                    ]
                    for item in arg_value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
                groups[box_id] = boxes.union(comments)
            elif all(
                isinstance(item, (ast.Name, ast.Subscript)) for item in arg_value.elts
            ):
                add_edge(arg_name, arg_value.elts, saved_values, ws, box_id, error_msg)
            else:
                list_value = []
                for item in arg_value.elts:
                    assert isinstance(item, ast.Constant), error_msg
                    list_value.append(item.value)
                params[arg_name] = list_value
        elif isinstance(arg_value, ast.Tuple):
            tuple_value = []
            for item in arg_value.elts:
                assert isinstance(item, ast.Constant), error_msg
                tuple_value.append(item.value)
            params[arg_name] = tuple(tuple_value)
    return params


def python_to_workspace(
    code: str, error_on_unknown_ops: bool = True
) -> workspace.Workspace:
    catalog = _get_catalog()
    tree = ast.parse(code)
    ws = workspace.Workspace()
    saved_values = {}
    comment_by_cleaned_text = {}
    # parse comments separately - they do not appear in the AST
    for line, text in workspace_comments.gather_multiline_comments(code):
        comment_by_cleaned_text[text.replace(" ", "").replace("\n", "")] = (
            f"comment on line {line}"
        )
        ws.add_node(
            id=f"comment on line {line}",
            title="Comment",
            op_id="Comment",
            type="comment",
            params={"text": text.strip()},
            width=400,
            height=50,
            position=workspace.Position(x=0, y=0),
        )
    groups = {}
    for s in tree.body:
        src = ast.get_source_segment(code, s)
        error_msg = f"Unexpected statement on line {s.lineno}:\n\n  {src}\n\nThe file must only contain function calls. Keyword arguments must be constants or previous results. Positional arguments are not allowed."
        save_as = None
        if (
            isinstance(s, ast.Import)
            and len(s.names) == 1
            and s.names[0].name == "boxes"
        ):
            # Ignore "import boxes".
            continue
        if isinstance(s, ast.Assign):
            assert len(s.targets) == 1, error_msg
            assert isinstance(s.targets[0], ast.Name), error_msg
            save_as = s.targets[0].id
        assert isinstance(s, (ast.Assign, ast.Expr)), error_msg
        s = s.value
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            # Ignore docstrings.
            continue
        assert isinstance(s, ast.Call), error_msg
        func_name = _get_func_name(s.func, error_msg)
        op_id = func_name
        box_id = f"{func_name} on line {s.lineno}"
        assert len(s.args) == 0, error_msg
        kwargs = {}
        for kw in s.keywords:
            assert kw.arg is not None, (
                f"{error_msg}\n\n**kwargs expansion is not supported."
            )
            kwargs[kw.arg] = kw.value
        params = parse_args(
            box_id,
            kwargs,
            saved_values,
            comment_by_cleaned_text,
            ws,
            error_msg,
            func_name,
            groups,
        )
        op = catalog.get(func_name)
        if op:
            box_title = op.name
            op_id = op.id
        elif func_name == "lynxkite_core.ops.group":
            op_id = "Group"
            box_title = "Group"
        elif error_on_unknown_ops:
            raise AssertionError(
                f"Unknown operation '{func_name}' on line {s.lineno}. "
                "Make sure the function is defined in boxes.py or is a pre-defined function."
            )
        else:
            box_title = func_name
        ws.add_node(
            id=box_id,
            title=box_title,
            op_id=op_id,
            params=params,
            width=400,
            height=400,
            type="node_group" if func_name == "lynxkite_core.ops.group" else "basic",
        )
        if save_as:
            saved_values[save_as] = box_id
    node_by_id = {node.id: node for node in ws.nodes}
    for group_id, node_ids in groups.items():
        for node_id in node_ids:
            node_by_id[node_id].parentId = group_id
    ws.update_metadata()
    return ws


def inputs_to_python(
    incoming_edges: list[workspace.WorkspaceEdge], saved_values: dict[str, str]
) -> list[str]:
    def handle_multi_output(edge: workspace.WorkspaceEdge) -> str:
        if edge.sourceHandle != "output":
            return f'{saved_values[edge.source]}["{edge.sourceHandle}"]'
        return saved_values[edge.source]

    sorted_incoming = sorted(incoming_edges, key=lambda e: e.targetHandle)
    grouped_by_target_handle_lists = [
        (th, [handle_multi_output(n) for n in gr])
        for th, gr in groupby(sorted_incoming, key=lambda e: e.targetHandle)
    ]
    grouped_by_target_handle = [
        (th, li[0] if len(li) == 1 else f"[{', '.join(li)}]")
        for th, li in grouped_by_target_handle_lists
    ]
    inputs = sorted(
        f"{targetHandle}={srcs}" for targetHandle, srcs in grouped_by_target_handle
    )
    return inputs


def _extract_info_from_edges(node_by_id, ws):
    incoming_edges: dict[str, list[workspace.WorkspaceEdge]] = {
        node.id: [] for node in ws.nodes
    }
    dependencies: dict[str, set[str]] = {node.id: set() for node in ws.nodes}
    for edge in ws.edges:
        # Ignore broken edges that point to missing nodes.
        if edge.source not in node_by_id or edge.target not in node_by_id:
            continue
        incoming_edges[edge.target].append(edge)
        dependencies[edge.target].add(edge.source)
    groups = {}
    for node in ws.nodes:
        if hasattr(node, "parentId") and node.parentId:
            dependencies[node.parentId].add(node.id)
            groups.setdefault(node.parentId, []).append(node.id)
    return incoming_edges, dependencies, groups


def _get_fnc_call(node, incoming_edges, saved_values, groups):
    inputs = inputs_to_python(incoming_edges[node.id], saved_values)
    params = sorted(f"{name}={repr(value)}" for name, value in node.data.params.items())
    if node.type == "node_group":
        params.append(
            f"group=[{', '.join(saved_values[n] for n in groups.get(node.id, []))}]"
        )
    meta = node.data.meta
    short_id = "".join(c if c.isalnum() else "_" for c in node.data.title.lower())
    if meta and meta.python_function_name:
        function_name = meta.python_function_name
    else:
        function_name = short_id
    call = f"{function_name}({', '.join(inputs + params)})  # {node.id}"
    return function_name, short_id, call


def _add_comment(node, code, saved_values):
    comment_lines = node.data.params.get("text", "").split("\n")
    saved_values[node.id] = (
        f'"{node.data.params.get("text", "").replace("\n", "\\n")}"'  # node.data.params.get("text", "")
    )
    for line in comment_lines:
        code.append(f"#! {line}")
    code.append("")


def workspace_to_python(ws: workspace.Workspace) -> str:
    code = [WORKSPACE_PROMPT]
    node_by_id = {node.id: node for node in ws.nodes}
    incoming_edges, dependencies, groups = _extract_info_from_edges(node_by_id, ws)
    sorter = graphlib.TopologicalSorter(dependencies)
    sorted_node_ids = list(sorter.static_order())
    saved_values: dict[str, str] = {}
    next_var_index = 1
    for node_id in sorted_node_ids:
        node = node_by_id[node_id]
        if node.type == "comment":
            _add_comment(node, code, saved_values)
            continue
        function_name, short_id, call = _get_fnc_call(
            node, incoming_edges, saved_values, groups
        )
        parent_op_metadata = [
            node_by_id[e.source].data.output_metadata
            for e in incoming_edges[node_id]
            if node_by_id[e.source].data.output_metadata
        ]
        input_comment, output_comment = workspace_comments.get_inp_op_metadata_comments(
            parent_op=reduce(lambda x, y: x + y, parent_op_metadata, []),
            ip=node.data.input_metadata if node.data.input_metadata else [],
            op=node.data.output_metadata if node.data.output_metadata else [],
        )
        if input_comment:
            code.append(f"\n# {function_name} {input_comment}")
        else:
            code.append("")
        variable_name = f"res_{short_id}_{next_var_index}"
        next_var_index += 1
        saved_values[node_id] = variable_name
        code.append(f"{variable_name} = {call}")
        if output_comment:
            code.append(f"# {function_name} {output_comment}")
    return "\n".join(code)
