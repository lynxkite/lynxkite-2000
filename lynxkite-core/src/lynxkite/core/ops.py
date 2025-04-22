"""API for implementing LynxKite operations."""

from __future__ import annotations
import asyncio
import enum
import functools
import importlib
import inspect
import pathlib
import subprocess
import traceback
import joblib
import types
import pydantic
import typing
from dataclasses import dataclass
from typing_extensions import Annotated

if typing.TYPE_CHECKING:
    from . import workspace

Catalog = dict[str, "Op"]
Catalogs = dict[str, Catalog]
CATALOGS: Catalogs = {}
EXECUTORS = {}
mem = joblib.Memory(".joblib-cache")

typeof = type  # We have some arguments called "type".


def type_to_json(t):
    if isinstance(t, type) and issubclass(t, enum.Enum):
        return {"enum": list(t.__members__.keys())}
    if getattr(t, "__metadata__", None):
        return t.__metadata__[-1]
    return {"type": str(t)}


Type = Annotated[typing.Any, pydantic.PlainSerializer(type_to_json, return_type=dict)]
LongStr = Annotated[str, {"format": "textarea"}]
PathStr = Annotated[str, {"format": "path"}]
CollapsedStr = Annotated[str, {"format": "collapsed"}]
NodeAttribute = Annotated[str, {"format": "node attribute"}]
EdgeAttribute = Annotated[str, {"format": "edge attribute"}]
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
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )


class Parameter(BaseConfig):
    """Defines a parameter for an operation."""

    name: str
    default: typing.Any
    type: Type = None

    @staticmethod
    def options(name, options, default=None):
        e = enum.Enum(f"OptionsFor_{name}", options)
        return Parameter.basic(name, default or options[0], e)

    @staticmethod
    def collapsed(name, default, type=None):
        return Parameter.basic(name, default, CollapsedStr)

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


class Input(BaseConfig):
    name: str
    type: Type
    # TODO: Make position an enum with the possible values.
    position: str = "left"


class Output(BaseConfig):
    name: str
    type: Type
    position: str = "right"


@dataclass
class Result:
    """Represents the result of an operation.

    The `output` attribute is what will be used as input for other operations.
    The `display` attribute is used to send data to display on the UI. The value has to be
    JSON-serializable.
    `input_metadata` is a list of JSON objects describing each input.
    """

    output: typing.Any = None
    display: ReadOnlyJSON | None = None
    error: str | None = None
    input_metadata: ReadOnlyJSON | None = None


MULTI_INPUT = Input(name="multi", type="*")


def basic_inputs(*names):
    return {name: Input(name=name, type=None) for name in names}


def basic_outputs(*names):
    return {name: Output(name=name, type=None) for name in names}


def _param_to_type(name, value, type):
    value = value or ""
    if type is int:
        assert value != "", f"{name} is unset."
        return int(value)
    if type is float:
        assert value != "", f"{name} is unset."
        return float(value)
    if isinstance(type, enum.EnumMeta):
        return type[value]
    if isinstance(type, types.UnionType):
        match type.__args__:
            case (types.NoneType, type):
                return None if value == "" else _param_to_type(name, value, type)
            case (type, types.NoneType):
                return None if value == "" else _param_to_type(name, value, type)
    if isinstance(type, typeof) and issubclass(type, pydantic.BaseModel):
        try:
            return type.model_validate_json(value)
        except pydantic.ValidationError:
            return None
    return value


class Op(BaseConfig):
    func: typing.Callable = pydantic.Field(exclude=True)
    name: str
    params: dict[str, Parameter | ParameterGroup]
    inputs: dict[str, Input]
    outputs: dict[str, Output]
    # TODO: Make type an enum with the possible values.
    type: str = "basic"  # The UI to use for this operation.

    def __call__(self, *inputs, **params):
        # Convert parameters.
        params = self.convert_params(params)
        res = self.func(*inputs, **params)
        if not isinstance(res, Result):
            # Automatically wrap the result in a Result object, if it isn't already.
            res = Result(output=res)
            if self.type in [
                "visualization",
                "table_view",
                "graph_creation_view",
                "image",
                "molecule",
            ]:
                # If the operation is some kind of visualization, we use the output as the
                # value to display by default.
                res.display = res.output
        return res

    def convert_params(self, params):
        """Returns the parameters converted to the expected type."""
        res = {}
        for p in params:
            if p in self.params:
                res[p] = _param_to_type(p, params[p], self.params[p].type)
            else:
                res[p] = params[p]
        return res


def op(env: str, name: str, *, view="basic", outputs=None, params=None, slow=False):
    """Decorator for defining an operation."""

    def decorator(func):
        sig = inspect.signature(func)
        if slow:
            func = mem.cache(func)
            func = _global_slow(func)
        # Positional arguments are inputs.
        inputs = {
            name: Input(name=name, type=param.annotation)
            for name, param in sig.parameters.items()
            if param.kind not in (param.KEYWORD_ONLY, param.VAR_KEYWORD)
        }
        _params = {}
        for n, param in sig.parameters.items():
            if param.kind == param.KEYWORD_ONLY and not n.startswith("_"):
                _params[n] = Parameter.basic(n, param.default, param.annotation)
        if params:
            _params.update(params)
        if outputs:
            _outputs = {name: Output(name=name, type=None) for name in outputs}
        else:
            _outputs = {"output": Output(name="output", type=None)} if view == "basic" else {}
        _view = view
        if view == "matplotlib":
            _view = "image"
            func = matplotlib_to_image(func)
        op = Op(
            func=func,
            name=name,
            params=_params,
            inputs=inputs,
            outputs=_outputs,
            type=_view,
        )
        CATALOGS.setdefault(env, {})
        CATALOGS[env][name] = op
        func.__op__ = op
        return func

    return decorator


def matplotlib_to_image(func):
    import matplotlib.pyplot as plt
    import base64
    import io

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    return wrapper


def input_position(**kwargs):
    """Decorator for specifying unusual positions for the inputs."""

    def decorator(func):
        op = func.__op__
        for k, v in kwargs.items():
            op.inputs[k].position = v
        return func

    return decorator


def output_position(**kwargs):
    """Decorator for specifying unusual positions for the outputs."""

    def decorator(func):
        op = func.__op__
        for k, v in kwargs.items():
            op.outputs[k].position = v
        return func

    return decorator


def no_op(*args, **kwargs):
    if args:
        return args[0]
    return None


def register_passive_op(env: str, name: str, inputs=[], outputs=["output"], params=[]):
    """A passive operation has no associated code."""
    op = Op(
        func=no_op,
        name=name,
        params={p.name: p for p in params},
        inputs=dict(
            (i, Input(name=i, type=None)) if isinstance(i, str) else (i.name, i) for i in inputs
        ),
        outputs=dict(
            (o, Output(name=o, type=None)) if isinstance(o, str) else (o.name, o) for o in outputs
        ),
    )
    CATALOGS.setdefault(env, {})
    CATALOGS[env][name] = op
    return op


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


def op_registration(env: str):
    """Returns a decorator that can be used for registering functions as operations."""
    return functools.partial(op, env)


def passive_op_registration(env: str):
    """Returns a function that can be used to register operations without associated code."""
    return functools.partial(register_passive_op, env)


def slow(func):
    """Decorator for slow, blocking operations. Turns them into separate threads."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


_global_slow = slow  # For access inside op().
CATALOGS_SNAPSHOTS: dict[str, Catalogs] = {}


def save_catalogs(snapshot_name: str):
    CATALOGS_SNAPSHOTS[snapshot_name] = {k: dict(v) for k, v in CATALOGS.items()}


def load_catalogs(snapshot_name: str):
    global CATALOGS
    snap = CATALOGS_SNAPSHOTS[snapshot_name]
    CATALOGS = {k: dict(v) for k, v in snap.items()}


def load_user_scripts(workspace: str):
    """Reloads the *.py in the workspace's directory and higher-level directories."""
    if "plugins loaded" in CATALOGS_SNAPSHOTS:
        load_catalogs("plugins loaded")
    cwd = pathlib.Path()
    path = cwd / workspace
    assert path.is_relative_to(cwd), "Provided workspace path is invalid"
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
        if p == cwd:
            break


def install_requirements(req: pathlib.Path):
    cmd = ["uv", "pip", "install", "-q", "-r", str(req)]
    subprocess.check_call(cmd)


def run_user_script(script_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
