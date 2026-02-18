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
    class Color(enum.IntEnum):
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


def test_detect_plugins_with_plugins():
    # This test assumes that these plugins are installed as part of the testing process.
    plugins = ops.detect_plugins()
    assert all(
        plugin in plugins
        for plugin in [
            "lynxkite_graph_analytics",
            "lynxkite_pillow_example",
        ]
    )


def test_pass_op_injects_context_and_message():
    observed_ctx = {}

    @ops.op("test_extra", "ctx", "pass-op", view="basic", outputs=["result"])
    def double_with_ctx(self, value: int):
        observed_ctx["ctx"] = self
        self.print("ctx message", append=False)
        return value * 2

    op = double_with_ctx.__op__
    result = op(3)

    assert observed_ctx["ctx"].op is op
    assert isinstance(result, ops.Result)
    assert result.output == 6
    assert result.message.strip() == "ctx message"


def test_cache_function_sync_and_async():
    call_count = {"n": 0}

    def sync_func(x):
        call_count["n"] += 1
        return x * 2

    async def async_func(x):
        call_count["n"] += 1
        return x * 2

    old_cache_wrapper = ops.CACHE_WRAPPER

    import joblib
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        mem = joblib.Memory(cache_dir, verbose=0)
        ops.CACHE_WRAPPER = mem.cache

        try:
            cached_sync = ops.cached(sync_func)
            cached_async = ops.cached(async_func)

            assert cached_sync(2) == 4
            assert cached_sync(2) == 4
            assert call_count["n"] == 1

            import asyncio

            async def test_async_cache():
                assert await cached_async(3) == 6
                assert await cached_async(3) == 6
                assert call_count["n"] == 2

            asyncio.run(test_async_cache())
        finally:
            ops.CACHE_WRAPPER = old_cache_wrapper


def test_cached_slow_op_with_message():
    call_count = {"n": 0}

    old_cache_wrapper = ops.CACHE_WRAPPER

    import joblib
    import tempfile
    import asyncio

    with tempfile.TemporaryDirectory() as cache_dir:
        mem = joblib.Memory(cache_dir, verbose=0)
        ops.CACHE_WRAPPER = mem.cache

        try:

            @ops.op("test_cache", "slow_add", view="basic", outputs=["result"], slow=True)
            def slow_add(self, a: int, b: int):
                call_count["n"] += 1
                self.print("computing")
                return a + b

            op = slow_add.__op__

            async def run():
                ctx1 = ops.OpContext(op=op)
                r1 = op(ctx1, 1, 2)
                r1.output = await r1.output
                r1 = ctx1.finalize_result_message(r1)
                assert r1.output == 3

                ctx2 = ops.OpContext(op=op)
                r2 = op(ctx2, 1, 2)
                r2.output = await r2.output
                r2 = ctx2.finalize_result_message(r2)
                assert r2.output == 3

                assert call_count["n"] == 1
                assert "computing" in (r2.message or "")

            asyncio.run(run())
        finally:
            ops.CACHE_WRAPPER = old_cache_wrapper
