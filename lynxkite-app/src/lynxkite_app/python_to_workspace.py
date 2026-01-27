"""
Convert a Python source file into a LynxKite workspace by extracting functions and their calls.
"""

import importlib.util
import ast
import typing
from lynxkite_core import ops, workspace
from lynxkite_core.executors import simple
from watchdog import events, observers

ENV = "Extracted from Python"
simple.register(ENV)
WATCHERS = {}


def load_module(source_path):
    spec = importlib.util.spec_from_file_location(source_path, source_path)
    assert spec
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def register_function(catalog: dict, source_path: str, mod, func_node: ast.FunctionDef) -> ops.Op:
    op_id = f"{source_path} > {func_node.name}"
    if op_id in catalog:
        return catalog[op_id]
    func = getattr(mod, func_node.name)
    op = ops.Op(
        func=func,
        doc=[func.__doc__ or ""],
        name=func_node.name,
        categories=[source_path],
        params=[],
        inputs=[],
        outputs=[ops.Output(name="output", type=typing.Any, position=ops.Position.RIGHT)],
        type="basic",
        color="orange",
        icon="python",
    )
    catalog[op_id] = op
    func.__op__ = op
    return op


class CodeConverter:
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.read_file()

    def read_file(self):
        with open(self.source_path, "r", encoding="utf-8") as f:
            self.source_code = f.read()
        tree = ast.parse(self.source_code, filename=self.source_path)
        mod = load_module(self.source_path)
        function_nodes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_nodes[node.name] = node
        ws = workspace.Workspace(env=ENV, path=self.source_path + ".lynxkite.json", paused=True)
        saved_values = {}
        box_x = {}
        box_y = {}
        catalog = {}
        assert "main" in function_nodes, "The source file must contain a main() function."
        main = function_nodes["main"]
        assert len(main.args.args) == 0, "The main() function must not take any arguments."
        for s in main.body:
            src = ast.get_source_segment(self.source_code, s)
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
            op = register_function(catalog, self.source_path, mod, func)
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
        self.ws = ws
        self.catalog = catalog

    def save_converted_source(self):
        env = "Extracted from Python"
        category = self.source_path.split("/")[-1].replace("_", " ").replace(".py", "")
        tree = ast.parse(self.source_code, filename=self.source_path)
        already_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                op_id = f"{self.source_path} > {node.name}"
                if op_id in self.catalog:
                    op = self.catalog[op_id]
                    inputs = {inp.name for inp in op.inputs}
                    first_default_index = len(node.args.args) - len(node.args.defaults)
                    new_args = []
                    new_defaults = []
                    for i, arg in enumerate(node.args.args):
                        if arg.arg in inputs:
                            new_args.append(arg)
                            if i >= first_default_index:
                                new_defaults.append(node.args.defaults[i - first_default_index])
                        else:
                            node.args.kwonlyargs.append(arg)
                            if i >= first_default_index:
                                node.args.kw_defaults.append(
                                    node.args.defaults[i - first_default_index]
                                )
                            else:
                                node.args.kw_defaults.append(None)
                    node.args.args = new_args
                    node.args.defaults = new_defaults
                    name = op.name.replace("_", " ")
                    decorator = ast.parse(f"op('{env}', '{category}', '{name}')").body[0]
                    assert isinstance(decorator, ast.Expr)
                    node.decorator_list.append(decorator.value)
            elif isinstance(node, ast.ImportFrom):
                if (
                    node.module == "lynxkite_core.ops"
                    and any(alias.name == "op" for alias in node.names)
                    and node.level == 0
                ):
                    already_imported = True
        if not already_imported:
            import_op = ast.parse("from lynxkite_core.ops import op").body[0]
            tree.body.insert(0, import_op)
        # TODO: Keep formatting at least inside function bodies.
        with open(self.source_path.replace(".py", "-converted.py"), "w") as f:
            f.write(ast.unparse(tree))

    def save_workspace(self):
        self.ws.save(self.source_path.replace(".py", ".lynxkite.json"))

    def watch_source(self):
        if self.source_path in WATCHERS:
            return
        event_handler = SourceChangeHandler(self)
        observer = observers.Observer()
        WATCHERS[self.source_path] = observer
        observer.schedule(event_handler, path=self.source_path, recursive=False)
        observer.start()


class SourceChangeHandler(events.FileSystemEventHandler):
    def __init__(self, converter: CodeConverter):
        self.converter = converter

    def on_modified(self, event):
        if event.src_path == self.converter.source_path:
            print(f"Detected changes in {event.src_path}. Updating workspace...")
            self.converter.read_file()
            self.converter.save_workspace()
            self.converter.save_converted_source()
