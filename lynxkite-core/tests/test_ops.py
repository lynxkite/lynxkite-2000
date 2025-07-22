import inspect
from lynxkite_core import ops
import enum


def test_op_decorator_no_params_no_types_default_positions():
    @ops.op("test", "add", view="basic", outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == []
    assert add.__op__.inputs == [
        ops.Input(name="a", type=inspect._empty, position=ops.Position.LEFT),
        ops.Input(name="b", type=inspect._empty, position=ops.Position.LEFT),
    ]
    assert add.__op__.outputs == [ops.Output(name="result", type=None, position=ops.Position.RIGHT)]
    assert add.__op__.type == "basic"
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_custom_positions():
    @ops.input_position(a="right", b="top")
    @ops.output_position(result="bottom")
    @ops.op("test", "add", view="basic", outputs=["result"])
    def add(a, b):
        return a + b

    assert add.__op__.name == "add"
    assert add.__op__.params == []
    assert add.__op__.inputs == [
        ops.Input(name="a", type=inspect._empty, position=ops.Position.RIGHT),
        ops.Input(name="b", type=inspect._empty, position=ops.Position.TOP),
    ]
    assert add.__op__.outputs == [
        ops.Output(name="result", type=None, position=ops.Position.BOTTOM)
    ]
    assert add.__op__.type == "basic"
    assert ops.CATALOGS["test"]["add"] == add.__op__


def test_op_decorator_with_params_and_types_():
    @ops.op("test", "multiply", view="basic", outputs=["result"])
    def multiply(a: int, b: float = 2.0, *, param: str = "param"):
        return a * b

    assert multiply.__op__.name == "multiply"
    assert multiply.__op__.params == [ops.Parameter(name="param", default="param", type=str)]
    assert multiply.__op__.inputs == [
        ops.Input(name="a", type=int, position=ops.Position.LEFT),
        ops.Input(name="b", type=float, position=ops.Position.LEFT),
    ]
    assert multiply.__op__.outputs == [
        ops.Output(name="result", type=None, position=ops.Position.RIGHT)
    ]
    assert multiply.__op__.type == "basic"
    assert ops.CATALOGS["test"]["multiply"] == multiply.__op__


def test_op_decorator_with_complex_types():
    class Color(int, enum.Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    @ops.op("test", "color_op", view="basic", outputs=["result"])
    def complex_op(color: Color, color_list: list[Color], color_dict: dict[str, Color]):
        return color.name

    assert complex_op.__op__.name == "color_op"
    assert complex_op.__op__.params == []
    assert complex_op.__op__.inputs == [
        ops.Input(name="color", type=Color, position=ops.Position.LEFT),
        ops.Input(name="color_list", type=list[Color], position=ops.Position.LEFT),
        ops.Input(name="color_dict", type=dict[str, Color], position=ops.Position.LEFT),
    ]
    assert complex_op.__op__.type == "basic"
    assert complex_op.__op__.outputs == [
        ops.Output(name="result", type=None, position=ops.Position.RIGHT)
    ]
    assert ops.CATALOGS["test"]["color_op"] == complex_op.__op__


def test_operation_can_return_non_result_instance():
    @ops.op("test", "subtract", view="basic", outputs=["result"])
    def subtract(a, b):
        return a - b

    result = ops.CATALOGS["test"]["subtract"](5, 3)
    assert isinstance(result, ops.Result)
    assert result.output == 2
    assert result.display is None


def test_operation_can_return_result_instance():
    @ops.op("test", "subtract", view="basic", outputs=["result"])
    def subtract(a, b):
        return ops.Result(output=a - b, display=None)

    result = ops.CATALOGS["test"]["subtract"](5, 3)
    assert isinstance(result, ops.Result)
    assert result.output == 2
    assert result.display is None


def test_visualization_operations_display_is_populated_automatically():
    @ops.op("test", "display_op", view="visualization", outputs=["result"])
    def display_op():
        return {"display_value": 1}

    result = ops.CATALOGS["test"]["display_op"]()
    assert isinstance(result, ops.Result)
    assert result.display == {"display_value": 1}
