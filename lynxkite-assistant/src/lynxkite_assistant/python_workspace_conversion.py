"""
Convert a series of Python function calls into a workspace, or the other way.
"""

import ast
import graphlib
from lynxkite_core import ops, workspace
from functools import reduce
from itertools import groupby


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


def _rec_convert(arg_value: ast.expr, error_msg: str):
    """Recursively converts the arg_value"""
    if isinstance(arg_value, ast.Constant):
        return arg_value.value
    if isinstance(arg_value, ast.List):
        return [_rec_convert(item, error_msg) for item in arg_value.elts]
    if isinstance(arg_value, ast.Tuple):
        return tuple(_rec_convert(item, error_msg) for item in arg_value.elts)
    raise AssertionError(error_msg)


def _gather_multiline_comments(code: str) -> list[tuple[int, str]]:
    comments = []
    comment_lines = []
    next_line = 0
    for i, line in enumerate(code.splitlines()):
        if not line.strip().startswith("#!"):
            continue
        if next_line == i:
            comment_lines.append(line.strip()[2:].strip())
        else:
            if comment_lines:
                comments.append(
                    (next_line - len(comment_lines), "\n".join(comment_lines))
                )
            comment_lines = [line.strip()[2:].strip()]
        next_line = i + 1
    if comment_lines:
        comments.append((next_line - len(comment_lines), "\n".join(comment_lines)))
    return comments


def python_to_workspace(
    code: str, error_on_unknown_ops: bool = True
) -> workspace.Workspace:
    catalog = _get_catalog()
    tree = ast.parse(code)
    ws = workspace.Workspace()
    saved_values = {}
    comment_by_cleaned_text = {}
    for line, text in _gather_multiline_comments(code):
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
        params = {}

        def add_edge(arg_name, arg_value_list):
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

        for arg_name, arg_value in kwargs.items():
            assert isinstance(
                arg_value,
                (ast.Constant, ast.Name, ast.Dict, ast.List, ast.Tuple, ast.Subscript),
            ), error_msg
            if isinstance(arg_value, ast.Constant):
                params[arg_name] = arg_value.value
            elif isinstance(arg_value, (ast.Name, ast.Subscript)):
                add_edge(arg_name, [arg_value])
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
                    isinstance(item, (ast.Name, ast.Constant))
                    for item in arg_value.elts
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
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    }
                    groups[box_id] = boxes.union(comments)
                elif all(
                    isinstance(item, (ast.Name, ast.Subscript))
                    for item in arg_value.elts
                ):
                    add_edge(arg_name, arg_value.elts)
                else:
                    params[arg_name] = _rec_convert(arg_value, error_msg)
            elif isinstance(arg_value, ast.Tuple):
                params[arg_name] = _rec_convert(arg_value, error_msg)
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


def describe_schema(schema: dict) -> str:
    lines = [f"bundle with {len(schema['dataframes'])} dataframe(s)"]
    for df_name, info in sorted(schema["dataframes"].items()):
        cols = ", ".join(f"'{c}'" for c in info.get("columns", []))
        lines.append(f"'{df_name}' with columns {cols}")
    return "\n# ".join(lines)


def compare_dataframe_schemas(prev_schema: dict, new_schema: dict) -> str:
    changes = []
    prev_keys = set(prev_schema["dataframes"].keys())
    new_keys = set(new_schema["dataframes"].keys())
    all_keys = prev_keys.union(new_keys)
    if prev_keys.intersection(new_keys) == set():
        return f"new bundle: {describe_schema(new_schema)}"
    for key in sorted(all_keys):
        if key not in prev_schema["dataframes"]:
            cols = ", ".join(f"'{c}'" for c in new_schema["dataframes"][key]["columns"])
            changes.append(f"new dataframe '{key}' added with columns: {cols}")
        elif key not in new_schema["dataframes"]:
            changes.append(f"dataframe '{key}' was removed")
        else:
            prev_cols = set(prev_schema["dataframes"][key]["columns"])
            new_cols = set(new_schema["dataframes"][key]["columns"])
            added_cols = sorted(list(new_cols - prev_cols))
            removed_cols = sorted(list(prev_cols - new_cols))
            if added_cols:
                cols_str = ", ".join(f"'{c}'" for c in added_cols)
                changes.append(f"'{key}' now has additional column(s): {cols_str}")
            if removed_cols:
                cols_str = ", ".join(f"'{c}'" for c in removed_cols)
                changes.append(f"'{key}' had column(s) removed: {cols_str}")
    return "\n# ".join(changes) if changes else "no change"


def get_inp_op_metadata_comments(
    parent_op: list[dict], ip: list[dict], op: list[dict]
) -> tuple[str, str]:
    def is_df_schema(d: dict) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        return "dataframes" in d and isinstance(d["dataframes"], dict)

    def get_compact_summary(meta: dict) -> str:
        if not meta:
            return ""
        keys_str = ", ".join(meta.keys())
        return f"metadata with keys: {keys_str}"

    input_descriptions = []
    # Filter out empty dicts from the lists completely
    parent_ops_clean = [p for p in parent_op if p]
    ips_clean = [i for i in ip if i]
    ops_clean = [o for o in op if o]
    for incoming in ips_clean:
        if incoming in parent_ops_clean:
            continue
        if is_df_schema(incoming):
            input_descriptions.append(f"input: {describe_schema(incoming)}")
        elif incoming:
            input_descriptions.append(f"input: {get_compact_summary(incoming)}")
    input_comment = "\n# ".join(input_descriptions) if input_descriptions else ""

    output_descriptions = []
    if ops_clean:
        primary_op = ops_clean[0]
        if is_df_schema(primary_op):
            # Find the most relevant baseline to compare against (input or parent output)
            baseline = next((i for i in ips_clean if is_df_schema(i)), None)
            if not baseline:
                baseline = next((p for p in parent_ops_clean if is_df_schema(p)), None)
            if baseline:
                output_descriptions.append(
                    "output: " + compare_dataframe_schemas(baseline, primary_op)
                )
            else:
                output_descriptions.append(
                    f"output: new bundle: {describe_schema(primary_op)}"
                )
        else:
            output_descriptions.append(f"output: {get_compact_summary(primary_op)}")
    output_comment = "\n# ".join(output_descriptions) if output_descriptions else ""
    return input_comment, output_comment


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


def workspace_to_python(ws: workspace.Workspace) -> str:
    code = [
        '"""The Python representation of the workspace."""',
        "# All imports are handled automatically. Do not add imports.",
        '# Only comments starting with "#!" are visible to the user and are interpreted as markdown. All other comments are for internal use only.',
        "# Comments not separated by a non-comment line are grouped together into a single comment. For example, the following two lines:",
        "#   #! This is a comment",
        "#   #! This is part of the same comment",
        "# will be grouped together into a single comment.",
        "# Therefore, if you want to create a new comment, make sure to add a non-comment line in between.",
        "# New user comments are placed above the next available function below it. (this rule doesn't apply to comments that are already in the workspace, they will stay where they are)",
        "# For example, if a comment is on line 1 and on line 3 and the next function is on line 4, the both comments will be placed above the box represented by the function on line 4.",
        "# Always place comments above the relevant line of code, so they appear above the box they are associated with.",
        "",
    ]
    node_by_id = {node.id: node for node in ws.nodes}
    incoming_edges: dict[str, list[workspace.WorkspaceEdge]] = {
        node.id: [] for node in ws.nodes
    }
    outgoing_count: dict[str, int] = {node.id: 0 for node in ws.nodes}
    dependencies: dict[str, set[str]] = {node.id: set() for node in ws.nodes}
    for edge in ws.edges:
        # Ignore broken edges that point to missing nodes.
        if edge.source not in node_by_id or edge.target not in node_by_id:
            continue
        incoming_edges[edge.target].append(edge)
        outgoing_count[edge.source] += 1
        dependencies[edge.target].add(edge.source)
    groups = {}
    for node in ws.nodes:
        if hasattr(node, "parentId") and node.parentId:
            dependencies[node.parentId].add(node.id)
            groups.setdefault(node.parentId, []).append(node.id)
    sorter = graphlib.TopologicalSorter(dependencies)
    sorted_node_ids = list(sorter.static_order())
    saved_values: dict[str, str] = {}
    next_var_index = 1
    for node_id in sorted_node_ids:
        node = node_by_id[node_id]
        if node.type == "comment":
            comment_lines = node.data.params.get("text", "").split("\n")
            saved_values[node_id] = (
                f'"{node.data.params.get("text", "").replace("\n", "\\n")}"'  # node.data.params.get("text", "")
            )
            for line in comment_lines:
                code.append(f"#! {line}")
            code.append("")
            continue
        inputs = inputs_to_python(incoming_edges[node_id], saved_values)
        params = sorted(
            f"{name}={repr(value)}" for name, value in node.data.params.items()
        )
        if node.type == "node_group":
            params.append(
                f"group=[{', '.join(saved_values[n] for n in groups.get(node_id, []))}]"
            )
        meta = node.data.meta
        short_id = "".join(c if c.isalnum() else "_" for c in node.data.title.lower())
        if meta and meta.python_function_name:
            function_name = meta.python_function_name
        else:
            function_name = short_id
        call = f"{function_name}({', '.join(inputs + params)})"
        parent_op_metadata = [
            node_by_id[e.source].data.output_metadata
            for e in incoming_edges[node_id]
            if node_by_id[e.source].data.output_metadata
        ]

        input_comment, output_comment = get_inp_op_metadata_comments(
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
