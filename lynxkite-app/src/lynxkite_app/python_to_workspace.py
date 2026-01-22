"""
Convert a Python source file into a LynxKite workspace by extracting functions and their calls.
"""

import importlib.util
import ast
import typing
from lynxkite_core import ops, workspace
from lynxkite_core.executors import simple

ENV = "Extracted from Python"
simple.register(ENV)


def load_module(source_path):
    spec = importlib.util.spec_from_file_location(source_path, source_path)
    assert spec
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def register_function(source_path, mod, func_node: ast.FunctionDef) -> ops.Op:
    ops.CATALOGS.setdefault(ENV, {})
    op_id = f"{source_path} > {func_node.name}"
    if op_id in ops.CATALOGS[ENV]:
        return ops.CATALOGS[ENV][op_id]
    func = getattr(mod, func_node.name)
    if func_node.name.startswith("plot"):
        outputs = []
        type = "image"
        func = ops.matplotlib_to_image(func)
        color = "blue"
    else:
        outputs = [ops.Output(name="output", type=typing.Any, position=ops.Position.RIGHT)]
        type = "basic"
        color = "green"
    op = ops.Op(
        func=func,
        doc=[func.__doc__ or ""],
        name=func_node.name,
        categories=[source_path],
        params=[],
        inputs=[],
        outputs=outputs,
        type=type,
        color=color,
        icon="brand-python",
    )
    ops.CATALOGS[ENV][op_id] = op
    func.__op__ = op
    return op


def code_as_workspace(source_path: str):
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=source_path)
    mod = load_module(source_path)
    function_nodes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_nodes[node.name] = node
    ws = workspace.Workspace(env=ENV, path=source_path + ".lynxkite.json", paused=True)
    saved_values = {}
    box_x = {}
    box_y = {}
    assert "main" in function_nodes, "The source file must contain a main() function."
    main = function_nodes["main"]
    assert len(main.args.args) == 0, "The main() function must not take any arguments."
    for s in main.body:
        src = ast.get_source_segment(source, s)
        error_msg = f"Unexpected statement on line {s.lineno}:\n\n  {src}\n\nThe main() function must only contain calls to other functions."
        save_as = None
        if isinstance(s, ast.Assign):
            assert len(s.targets) == 1, error_msg
            assert isinstance(s.targets[0], ast.Name), error_msg
            save_as = s.targets[0].id
        assert isinstance(s, ast.Assign | ast.Expr), error_msg
        s = s.value
        assert isinstance(s, ast.Call), error_msg
        assert isinstance(s.func, ast.Name), error_msg
        assert s.func.id in function_nodes, error_msg
        box_id = f"{s.func.id} on line {s.lineno}"
        func = function_nodes[s.func.id]
        op = register_function(source_path, mod, func)
        func_args = [a.arg for a in func.args.args]
        kwargs = {}
        assert len(s.args) <= len(func_args), error_msg
        for arg_node, arg_name in zip(s.args, func_args):
            kwargs[arg_name] = arg_node
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
                matches = [p for p in op.params if p.name == arg_name]
                if not matches:
                    op.params.append(
                        ops.Parameter(
                            name=arg_name, type=type(arg_value.value), default=arg_value.value
                        )
                    )
            elif isinstance(arg_value, ast.Name):
                assert arg_value.id in saved_values, error_msg
                op.color = "orange" if op.color == "green" else op.color
                matches = [i for i in op.inputs if i.name == arg_name]
                if not matches:
                    op.inputs.append(
                        ops.Input(name=arg_name, type=typing.Any, position=ops.Position.LEFT)
                    )
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
            title=func.name,
            op_id=op.id,
            params=params,
            width=300,
            height=200,
            position=workspace.Position(x=x * 400, y=y * 250),
        )
        if save_as:
            saved_values[save_as] = box_id
    ws.update_metadata()
    return ws
