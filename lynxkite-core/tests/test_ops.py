import inspect
from lynxkite.core import ops
import enum


def test_op_decorator_no_params_no_types_default_sides():
    @ops.op(env="test", name="add", view=ops.ViewType.BASIC, outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == {}
    assert add.__op__.inputs == {
        "a": ops.Input(name="a", type=inspect._empty, side=ops.Side.LEFT),
        "b": ops.Input(name="b", type=inspect._empty, side=ops.Side.LEFT),
    }
    assert add.__op__.outputs == {
        "result": ops.Output(name="result", type=None, side=ops.Side.RIGHT)
    }
    assert add.__op__.view_type == ops.ViewType.BASIC
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_custom_sides():
    @ops.input_side(a=ops.Side.RIGHT, b=ops.Side.TOP)
    @ops.output_side(result=ops.Side.BOTTOM)
    @ops.op(env="test", name="add", view=ops.ViewType.BASIC, outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == {}
    assert add.__op__.inputs == {
        "a": ops.Input(name="a", type=inspect._empty, side=ops.Side.RIGHT),
        "b": ops.Input(name="b", type=inspect._empty, side=ops.Side.TOP),
    }
    assert add.__op__.outputs == {
        "result": ops.Output(name="result", type=None, side=ops.Side.BOTTOM)
    }
    assert add.__op__.view_type == ops.ViewType.BASIC
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_with_params_and_types_():
    @ops.op(env="test", name="multiply", view=ops.ViewType.BASIC, outputs=["result"])
    def multiply(a: int, b: float = 2.0, *, param: str = "param"):
        return a * b

    assert multiply.__op__.name == "multiply"
    assert multiply.__op__.params == {
        "param": ops.Parameter(name="param", default="param", type=str)
    }
    assert multiply.__op__.inputs == {
        "a": ops.Input(name="a", type=int, side=ops.Side.LEFT),
        "b": ops.Input(name="b", type=float, side=ops.Side.LEFT),
    }
    assert multiply.__op__.outputs == {
        "result": ops.Output(name="result", type=None, side=ops.Side.RIGHT)
    }
    assert multiply.__op__.view_type == ops.ViewType.BASIC
    assert ops.CATALOGS["test"]["multiply"] == multiply.__op__


def test_op_decorator_with_complex_types():
    class Color(enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    @ops.op(env="test", name="color_op", view=ops.ViewType.BASIC, outputs=["result"])
    def complex_op(color: Color, color_list: list[Color], color_dict: dict[str, Color]):
        return color.name

    assert complex_op.__op__.name == "color_op"
    assert complex_op.__op__.params == {}
    assert complex_op.__op__.inputs == {
        "color": ops.Input(name="color", type=Color, side=ops.Side.LEFT),
        "color_list": ops.Input(
            name="color_list", type=list[Color], side=ops.Side.LEFT
        ),
        "color_dict": ops.Input(
            name="color_dict", type=dict[str, Color], side=ops.Side.LEFT
        ),
    }
    assert complex_op.__op__.view_type == ops.ViewType.BASIC
    assert complex_op.__op__.outputs == {
        "result": ops.Output(name="result", type=None, side=ops.Side.RIGHT)
    }
    assert ops.CATALOGS["test"]["color_op"] == complex_op.__op__
