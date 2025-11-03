"""API for implementing LynxKite operations."""

from __future__ import annotations

import asyncio
import enum
import functools
import json
import importlib.util
import inspect
import pathlib
import pkgutil
import subprocess
import traceback
import types
import typing
from dataclasses import dataclass

import pydantic

if typing.TYPE_CHECKING:
    from . import workspace

Catalog = dict[str, "Op"]
Catalogs = dict[str, Catalog]
CATALOGS: Catalogs = {}
EXECUTORS = {}

typeof = type  # We have some arguments called "type".

CACHE_WRAPPER = None  # Overwrite this to configure a caching mechanism.


def cached(func):
    if CACHE_WRAPPER is None:
        return func
    return CACHE_WRAPPER(func)


def type_to_json(t):
    if isinstance(t, type) and issubclass(t, enum.Enum):
        return {"enum": list(t.__members__.values())}
    if getattr(t, "__metadata__", None):
        return t.__metadata__[-1]
    return {"type": str(t)}


Type = typing.Annotated[typing.Any, pydantic.PlainSerializer(type_to_json, return_type=dict)]
LongStr = typing.Annotated[str, {"format": "textarea"}]
"""LongStr is a string type for parameters that will be displayed as a multiline text area in the UI."""
PathStr = typing.Annotated[str, {"format": "path"}]
# https://github.com/python/typing/issues/182#issuecomment-1320974824
ReadOnlyJSON: typing.TypeAlias = (
    typing.Mapping[str, "ReadOnlyJSON"]
    | typing.Sequence["ReadOnlyJSON"]
    | str
    | int
    | float
    | bool
    | None
)


class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict()


class Parameter(BaseConfig):
    """Defines a parameter for an operation."""

    name: str
    default: typing.Any
    type: Type = None

    @staticmethod
    def options(name, options: list[str], default=None):
        e = enum.StrEnum(f"OptionsFor_{name}", [(o, o) for o in options])
        return Parameter.basic(name, default or options[0], e)

    @staticmethod
    def basic(name, default=None, type=None):
        if default is inspect._empty:
            default = None
        if type is None or type is inspect._empty:
            type = typeof(default) if default is not None else None
        return Parameter(name=name, default=default, type=type)


class ParameterGroup(BaseConfig):
    """Defines a group of parameters for an operation."""

    name: str
    selector: Parameter
    default: typing.Any
    groups: dict[str, list[Parameter]]
    type: str = "group"


class Position(enum.StrEnum):
    """Defines the position of an input or output in the UI."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    def is_vertical(self):
        return self in (self.TOP, self.BOTTOM)

    @staticmethod
    def from_dir(dir: str) -> tuple[Position, Position]:
        """Returns the input and output positions based on the direction."""
        if dir == "left-to-right":
            return Position.LEFT, Position.RIGHT
        elif dir == "right-to-left":
            return Position.RIGHT, Position.LEFT
        elif dir == "top-to-bottom":
            return Position.TOP, Position.BOTTOM
        elif dir == "bottom-to-top":
            return Position.BOTTOM, Position.TOP
        else:
            raise ValueError(f"Invalid direction: {dir}")


class Input(BaseConfig):
    name: str
    type: Type
    position: Position


class Output(BaseConfig):
    name: str
    type: Type
    position: Position


@dataclass
class Result:
    """Represents the result of an operation.

    The `output` attribute is what will be used as input for other operations.
    The `display` attribute is used to send data to display on the UI. The value has to be
    JSON-serializable.
    `input_metadata` is a list of JSON objects describing each input.
    """

    output: typing.Any | None = None
    display: ReadOnlyJSON | None = None
    error: str | None = None
    input_metadata: list[dict[str, ReadOnlyJSON]] | None = None


def get_optional_type(type):
    """For a type like `int | None`, returns `int`. Returns `None` otherwise."""
    if isinstance(type, types.UnionType):
        match type.__args__:
            case (types.NoneType, type):
                return type
            case (type, types.NoneType):
                return type


def _param_to_type(name, value, type):
    if value is None:
        value = ""
    if type is int:
        assert value != "", f"{name} is unset."
        return int(value)
    if type is float:
        assert value != "", f"{name} is unset."
        return float(value)
    if isinstance(type, enum.EnumMeta):
        assert value in type, (
            f'Parameter "{name}" must be one of {[t.value for t in type]}. Found: {value}'
        )
        return type(value)
    opt_type = get_optional_type(type)
    if opt_type:
        return None if value == "" else _param_to_type(name, value, opt_type)
    if isinstance(type, typeof) and issubclass(type, pydantic.BaseModel):
        return type.model_validate_json(value)
    return value


class Op(BaseConfig):
    func: typing.Callable = pydantic.Field(exclude=True)
    categories: list[str]
    name: str
    params: list[Parameter | ParameterGroup]
    inputs: list[Input]
    outputs: list[Output]
    # TODO: Make type an enum with the possible values.
    type: str = "basic"  # The UI to use for this operation.
    color: str = "orange"  # The color of the operation in the UI.
    doc: list | None = None
    # ID is automatically set from the name and categories.
    id: str = pydantic.Field(default=None)

    def __call__(self, *inputs, **params):
        # Convert parameters.
        params = self.convert_params(params)
        res = self.func(*inputs, **params)
        if not isinstance(res, Result):
            # Automatically wrap the result in a Result object, if it isn't already.
            if self.type in [
                "visualization",
                "table_view",
                "graph_creation_view",
                "image",
                "molecule",
            ]:
                # If the operation is a visualization, we use the returned value for display.
                res = Result(display=res)
            else:
                res = Result(output=res)
        return res

    def get_input(self, name: str):
        """Returns the input with the given name."""
        for i in self.inputs:
            if i.name == name:
                return i
        raise ValueError(f"Input {name} not found in operation {self.name}.")

    def get_output(self, name: str):
        """Returns the output with the given name."""
        for o in self.outputs:
            if o.name == name:
                return o
        raise ValueError(f"Output {name} not found in operation {self.name}.")

    def convert_params(self, params: dict[str, typing.Any]):
        """Returns the parameters converted to the expected type."""
        res = dict(params)
        for p in self.params:
            if p.name in params:
                res[p.name] = _param_to_type(p.name, params[p.name], p.type)
        return res

    @pydantic.model_validator(mode="after")
    def compute_id(self):
        self.id = " > ".join(self.categories + [self.name])
        return self


def op(
    env: str | None,
    *names: str,
    view: str = "basic",
    outputs: list[str] | None = None,
    params: list[Parameter] | None = None,
    slow: bool = False,
    color: str | None = None,
    cache: bool | None = None,
    dir: str = "left-to-right",
):
    """
    Decorator for defining an operation.

    Parameters:
        env: The environment (workspace type) to which the operation belongs. If None, the operation
             is not registered in any catalog.
        names: The list of categories this operation belongs to, followed by the name of the operation.
        view: How the operation will be displayed in the UI. One of "basic", "visualization",
              "table_view", "graph_creation_view", "image", "molecule", "matplotlib", "service".
        outputs: A list of output names. If not provided, defaults to ["output"] for "basic" view.
        params: Normally the parameters are taken from the function signature.
                Use "params" to override this.
        slow: If True, the operation results will be cached.
              If the function is not async, it will be run in a separate thread.
        color: The color of the operation in the UI. Defaults to "orange".
        cache: Set to False to disable caching for a slow operation.
               You may need this for slow operations with parameters/outputs that can't be serialized.
        dir: Sets the default input and output positions. The default is "left-to-right", meaning
             inputs are on the left and outputs are on the right. Other options are "right-to-left",
             "top-to-bottom", and "bottom-to-top".
    """
    [*categories, name] = names

    def decorator(func):
        doc = parse_doc(func)
        sig = inspect.signature(func)
        _view = view
        if view == "matplotlib":
            _view = "image"
            func = matplotlib_to_image(func)
        if slow:
            func = make_async(func)
            if cache is not False:
                func = cached(func)
        # Positional arguments are inputs.
        ipos, opos = Position.from_dir(dir)
        inputs = [
            Input(name=name, type=param.annotation, position=ipos)
            for name, param in sig.parameters.items()
            if param.kind not in (param.KEYWORD_ONLY, param.VAR_KEYWORD)
        ]
        _params = []
        for n, param in sig.parameters.items():
            if param.kind == param.KEYWORD_ONLY and not n.startswith("_"):
                _params.append(Parameter.basic(n, param.default, param.annotation))
        if params:
            _params.extend(params)
        if outputs is not None:
            _outputs = [Output(name=name, type=None, position=opos) for name in outputs]
        else:
            _outputs = [Output(name="output", type=None, position=opos)] if view == "basic" else []
        op = Op(
            func=func,
            doc=doc,
            name=name,
            categories=categories,
            params=_params,
            inputs=inputs,
            outputs=_outputs,
            type=_view,
            color=color or "orange",
        )
        if env is not None:
            CATALOGS.setdefault(env, {})
            CATALOGS[env][op.id] = op
        func.__op__ = op
        return func

    return decorator


def matplotlib_to_image(func):
    """Decorator for converting a matplotlib figure to an image."""
    import base64
    import io

    import matplotlib.pyplot as plt
    import matplotlib

    # Make sure we use the non-interactive backend.
    matplotlib.use("agg")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    return wrapper


def input_position(**positions):
    """
    Decorator for specifying unusual positions for the inputs.

    Example usage:

        @input_position(a="bottom", b="bottom")
        @op("test", "maybe add")
        def maybe_add(a: list[int], b: list[int] | None = None):
            return [a + b for a, b in zip(a, b)] if b else a
    """

    def decorator(func):
        op = func.__op__
        for k, v in positions.items():
            op.get_input(k).position = Position(v)
        return func

    return decorator


def output_position(**positions):
    """Decorator for specifying unusual positions for the outputs.

    Example usage:

        @output_position(output="top")
        @op("test", "maybe add")
        def maybe_add(a: list[int], b: list[int] | None = None):
            return [a + b for a, b in zip(a, b)] if b else a
    """

    def decorator(func):
        op = func.__op__
        for k, v in positions.items():
            op.get_output(k).position = Position(v)
        return func

    return decorator


def no_op(*args, **kwargs):
    if args:
        return args[0]
    return None


def register_passive_op(
    env: str, *names: str, inputs=[], outputs=["output"], params=[], dir="left-to-right", **kwargs
):
    """A passive operation has no associated code."""
    ipos, opos = Position.from_dir(dir)
    [*categories, name] = names
    op = Op(
        func=no_op,
        name=name,
        categories=categories,
        params=params,
        inputs=[
            Input(name=i, type=None, position=ipos) if isinstance(i, str) else i for i in inputs
        ],
        outputs=[
            Output(name=o, type=None, position=opos) if isinstance(o, str) else o for o in outputs
        ],
        **kwargs,
    )
    CATALOGS.setdefault(env, {})
    CATALOGS[env][op.id] = op
    return op


COMMENT_OP = Op(
    func=no_op,
    name="Comment",
    categories=[],
    params=[Parameter.basic("text", "", LongStr)],
    inputs=[],
    outputs=[],
    type="comment",
)


def register_executor(env: str):
    """Decorator for registering an executor.

    The executor is a function that takes a workspace and executes the operations in it.
    When it starts executing an operation, it should call `node.publish_started()` to indicate
    the status on the UI. When the execution is finished, it should call `node.publish_result()`.
    This will update the UI with the result of the operation.
    """

    def decorator(func: typing.Callable[[workspace.Workspace], typing.Any]):
        EXECUTORS[env] = func
        return func

    return decorator


def op_registration(env: str, *categories: str, **kwargs):
    """Returns a decorator that can be used for registering functions as operations."""
    return functools.partial(op, env, *categories, **kwargs)


def passive_op_registration(env: str, *categories: str, **kwargs):
    """Returns a function that can be used to register operations without associated code."""
    return functools.partial(register_passive_op, env, *categories, **kwargs)


def make_async(func):
    """Decorator for slow, blocking operations. Turns them into separate threads."""

    if asyncio.iscoroutinefunction(func):
        # If the function is already a coroutine, return it as is.
        return func

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


CATALOGS_SNAPSHOTS: dict[str, Catalogs] = {}


def save_catalogs(snapshot_name: str):
    CATALOGS_SNAPSHOTS[snapshot_name] = {k: dict(v) for k, v in CATALOGS.items()}


def load_catalogs(snapshot_name: str):
    global CATALOGS
    snap = CATALOGS_SNAPSHOTS[snapshot_name]
    CATALOGS = {k: dict(v) for k, v in snap.items()}


# Generally the same as the data directory, but it can be overridden.
user_script_root = pathlib.Path()


def load_user_scripts(workspace: str):
    """Reloads the *.py in the workspace's directory and higher-level directories."""
    if "plugins loaded" in CATALOGS_SNAPSHOTS:
        load_catalogs("plugins loaded")
    if not user_script_root:
        return
    path = user_script_root / workspace
    assert path.is_relative_to(user_script_root), f"Path '{path}' is invalid"
    for p in path.parents:
        req = p / "requirements.txt"
        if req.exists():
            try:
                install_requirements(req)
            except Exception:
                traceback.print_exc()
        for f in p.glob("*.py"):
            try:
                run_user_script(f)
            except Exception:
                traceback.print_exc()
        if p == user_script_root:
            break


def install_requirements(req: pathlib.Path):
    cmd = ["uv", "pip", "install", "-q", "-r", str(req)]
    subprocess.check_call(cmd)


def run_user_script(script_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    assert spec
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)


@functools.cache
def parse_doc(func) -> list | None:
    """Griffe is an optional dependency. When available, we return the parsed docstring."""
    doc = func.__doc__
    try:
        import griffe
    except ImportError:
        return None
    if doc is None:
        return None
    griffe.logger.setLevel("ERROR")
    ds = griffe.Docstring(doc, parent=_get_griffe_function(func))
    if "----" in doc:
        ds = ds.parse("numpy")
    else:
        ds = ds.parse("google")
    return json.loads(json.dumps(ds, cls=griffe.JSONEncoder))


def _get_griffe_function(func):
    """Returns a griffe.Function object based on the signature of the function."""
    import griffe

    sig = inspect.signature(func)
    parameters = []
    for name, param in sig.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            annotation = None
        else:
            annotation = getattr(param.annotation, "__name__", str(param.annotation))
        parameters.append(
            griffe.Parameter(
                name,
                annotation=annotation,
                kind=griffe.ParameterKind.keyword_only
                if param.kind == inspect.Parameter.KEYWORD_ONLY
                else griffe.ParameterKind.positional_or_keyword,
            )
        )
    return griffe.Function(
        func.__name__,
        parameters=griffe.Parameters(*parameters),
        returns=str(sig.return_annotation),
    )


def detect_plugins():
    """Imports all installed LynxKite plugins."""
    plugins = {}
    for _, name, _ in pkgutil.iter_modules():
        if (
            name.startswith("lynxkite_")
            and name != "lynxkite_app"
            and name != "lynxkite_core"
            and name != "lynxkite_mcp"
        ):
            plugins[name] = importlib.import_module(name)
    if not plugins:
        print("No LynxKite plugins found. Be sure to install some!")
    return plugins
