import pytest
from lynxkite_core import ops

DummyContext = ops.OpContext(
    op=None
)  # For testing the op decorator without needing a full execution context.


def test_annotated_opcontext_excluded_from_inputs():
    """A parameter annotated as OpContext is excluded from graph inputs."""

    @ops.op("test", "annotated_ctx_op", outputs=["output"])
    def annotated_ctx_op(op_ctx: ops.OpContext, data: int, *, label: str = "x"):
        op_ctx.set_message(f"got {data} label={label}")
        return data

    op_obj = annotated_ctx_op.__op__
    input_names = [i.name for i in op_obj.inputs]
    assert "op_ctx" not in input_names, "OpContext param must not appear as a graph input"
    assert "data" in input_names


def test_annotated_opcontext_injected_at_runtime():
    """Annotation-based OpContext is injected as keyword arg at call time."""

    received = {}

    @ops.op("test", "annotated_ctx_inject", outputs=["output"])
    def annotated_ctx_inject(op_ctx: ops.OpContext, value: int):
        received["ctx"] = op_ctx
        return value

    result = annotated_ctx_inject.__op__(DummyContext, 7)
    assert result.output == 7
    assert received["ctx"] is DummyContext


def test_annotated_opcontext_in_any_position():
    """OpContext can be annotated in any position, and is injected at runtime."""

    received = {}

    @ops.op("test", "annotated_ctx_any_pos", outputs=["output"])
    def annotated_ctx_any_pos(value: int, op_ctx: ops.OpContext, *, label: str = "x"):
        received["ctx"] = op_ctx
        return value

    result = annotated_ctx_any_pos.__op__(DummyContext, 42, label="y")
    assert result.output == 42
    assert received["ctx"] is DummyContext


def test_annotated_opcontext_in_kw_args_is_rejected():
    """OpContext annotated as keyword-only should fail fast at declaration time."""

    with pytest.raises(ValueError, match="keyword-only"):

        @ops.op("test", "annotated_ctx_in_kw_args", outputs=["output"])
        def annotated_ctx_in_kw_args(value: int, *, op_ctx: ops.OpContext, label: str = "x"):
            return value


@pytest.mark.asyncio
async def test_async_opcontext_in_any_position():
    received = {}

    @ops.op("test", "async_ctx_any_pos", outputs=["output"])
    async def async_no_ctx(input: int, op_ctx: ops.OpContext, *, table: str):
        received["ctx"] = op_ctx
        return input * 2

    result = await async_no_ctx.__op__(DummyContext, 7, table="rows").output
    assert result == 14
    assert received["ctx"] is DummyContext
