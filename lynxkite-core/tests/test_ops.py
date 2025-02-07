import inspect
from lynxkite.core import ops
import enum


def test_op_decorator_no_params_no_types_default_positions():
    @ops.op(env="test", name="add", view="basic", outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == {}
    assert add.__op__.inputs == {
        "a": ops.Input(name="a", type=inspect._empty, position="left"),
        "b": ops.Input(name="b", type=inspect._empty, position="left"),
    }
    assert add.__op__.outputs == {
        "result": ops.Output(name="result", type=None, position="right")
    }
    assert add.__op__.type == "basic"
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_custom_positions():
    @ops.input_position(a="right", b="top")
    @ops.output_position(result="bottom")
    @ops.op(env="test", name="add", view="basic", outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == {}
    assert add.__op__.inputs == {
        "a": ops.Input(name="a", type=inspect._empty, position="right"),
        "b": ops.Input(name="b", type=inspect._empty, position="top"),
    }
    assert add.__op__.outputs == {
        "result": ops.Output(name="result", type=None, position="bottom")
    }
    assert add.__op__.type == "basic"
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_with_params_and_types_():
    @ops.op(env="test", name="multiply", view="basic", outputs=["result"])
    def multiply(a: int, b: float = 2.0, *, param: str = "param"):
        return a * b

    assert multiply.__op__.name == "multiply"
    assert multiply.__op__.params == {
        "param": ops.Parameter(name="param", default="param", type=str)
    }
    assert multiply.__op__.inputs == {
        "a": ops.Input(name="a", type=int, position="left"),
        "b": ops.Input(name="b", type=float, position="left"),
    }
    assert multiply.__op__.outputs == {
        "result": ops.Output(name="result", type=None, position="right")
    }
    assert multiply.__op__.type == "basic"
    assert ops.CATALOGS["test"]["multiply"] == multiply.__op__


def test_op_decorator_with_complex_types():
    class Color(enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    @ops.op(env="test", name="color_op", view="basic", outputs=["result"])
    def complex_op(color: Color, color_list: list[Color], color_dict: dict[str, Color]):
        return color.name

    assert complex_op.__op__.name == "color_op"
    assert complex_op.__op__.params == {}
    assert complex_op.__op__.inputs == {
        "color": ops.Input(name="color", type=Color, position="left"),
        "color_list": ops.Input(name="color_list", type=list[Color], position="left"),
        "color_dict": ops.Input(name="color_dict", type=dict[str, Color], position="left"),
    }
    assert complex_op.__op__.type == "basic"
    assert complex_op.__op__.outputs == {
        "result": ops.Output(name="result", type=None, position="right")
    }
    assert ops.CATALOGS["test"]["color_op"] == complex_op.__op__
