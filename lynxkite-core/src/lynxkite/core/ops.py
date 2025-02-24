"""API for implementing LynxKite operations."""

from __future__ import annotations
import enum
import functools
import inspect
import pydantic
import typing
from dataclasses import dataclass
from typing_extensions import Annotated

CATALOGS = {}
EXECUTORS = {}

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
        return Parameter.basic(name, e[default or options[0]], e)

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
    """

    output: typing.Any
    display: ReadOnlyJSON | None = None


MULTI_INPUT = Input(name="multi", type="*")


def basic_inputs(*names):
    return {name: Input(name=name, type=None) for name in names}


def basic_outputs(*names):
    return {name: Output(name=name, type=None) for name in names}


class Op(BaseConfig):
    func: typing.Callable = pydantic.Field(exclude=True)
    name: str
    params: dict[str, Parameter]
    inputs: dict[str, Input]
    outputs: dict[str, Output]
    # TODO: Make type an enum with the possible values.
    type: str = "basic"  # The UI to use for this operation.

    def __call__(self, *inputs, **params):
        # Convert parameters.
        for p in params:
            if p in self.params:
                if self.params[p].type is int:
                    params[p] = int(params[p])
                elif self.params[p].type is float:
                    params[p] = float(params[p])
                elif isinstance(self.params[p].type, enum.EnumMeta):
                    params[p] = self.params[p].type[params[p]]
        res = self.func(*inputs, **params)
        if not isinstance(res, Result):
            # Automatically wrap the result in a Result object, if it isn't already.
            res = Result(output=res)
            if self.type in [
                "visualization",
                "table_view",
                "graph_creation_view",
                "image",
            ]:
                # If the operation is some kind of visualization, we use the output as the
                # value to display by default.
                res.display = res.output
        return res


def op(env: str, name: str, *, view="basic", outputs=None):
    """Decorator for defining an operation."""

    def decorator(func):
        sig = inspect.signature(func)
        # Positional arguments are inputs.
        inputs = {
            name: Input(name=name, type=param.annotation)
            for name, param in sig.parameters.items()
            if param.kind != param.KEYWORD_ONLY
        }
        params = {}
        for n, param in sig.parameters.items():
            if param.kind == param.KEYWORD_ONLY and not n.startswith("_"):
                params[n] = Parameter.basic(n, param.default, param.annotation)
        if outputs:
            _outputs = {name: Output(name=name, type=None) for name in outputs}
        else:
            _outputs = (
                {"output": Output(name="output", type=None)} if view == "basic" else {}
            )
        op = Op(
            func=func,
            name=name,
            params=params,
            inputs=inputs,
            outputs=_outputs,
            type=view,
        )
        CATALOGS.setdefault(env, {})
        CATALOGS[env][name] = op
        func.__op__ = op
        return func

    return decorator


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
            (i, Input(name=i, type=None)) if isinstance(i, str) else (i.name, i)
            for i in inputs
        ),
        outputs=dict(
            (o, Output(name=o, type=None)) if isinstance(o, str) else (o.name, o)
            for o in outputs
        ),
    )
    CATALOGS.setdefault(env, {})
    CATALOGS[env][name] = op
    return op


def register_executor(env: str):
    """Decorator for registering an executor."""

    def decorator(func):
        EXECUTORS[env] = func
        return func

    return decorator


def op_registration(env: str):
    """Returns a decorator that can be used for registering functions as operations."""
    return functools.partial(op, env)


def passive_op_registration(env: str):
    """Returns a function that can be used to register operations without associated code."""
    return functools.partial(register_passive_op, env)
