"""Regression tests for OpContext injection in @elementwise row functions."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from lynxkite_core.opcontext import OpContext
from lynxkite_core.ops import op
from lynxkite_graph_analytics.bundle import Bundle
from lynxkite_graph_analytics.elementwise import elementwise
from lynxkite_graph_analytics.record import Record

ENV = "test_elementwise_opctx"


def _make_bundle(*labels: str) -> Bundle:
    df = pd.DataFrame({"label": list(labels)})
    return Bundle(dfs={"rows": df})


DummyContext = OpContext(
    op=None
)  # For testing the op decorator without needing a full execution context.


def test_elementwise_opctx_hidden_from_public_signature():
    @op(ENV, "Tag rows hidden sig")
    @elementwise(input_table="table")
    def tag(record: Record, op_ctx: OpContext, *, table: str, tag: str = "t") -> None:
        pass

    sig = inspect.signature(tag)
    assert "op_ctx" not in sig.parameters, "op_ctx must not appear in the public wrapper signature"
    assert "table" in sig.parameters
    assert "tag" in sig.parameters


def test_elementwise_without_opctx_signature_unchanged():
    @op(ENV, "Tag rows no ctx sig")
    @elementwise(input_table="table")
    def tag_no_ctx(record: Record, *, table: str, tag: str = "t") -> None:
        pass

    sig = inspect.signature(tag_no_ctx)
    assert "table" in sig.parameters
    assert "tag" in sig.parameters


def test_sync_elementwise_injects_opcontext():
    injected: list[OpContext] = []

    @op(ENV, "Sync inject")
    @elementwise(input_table="table")
    def sync_row(record: Record, op_ctx: OpContext, *, table: str) -> None:
        injected.append(op_ctx)
        record["label"] = record["label"] + "_done"

    bundle = _make_bundle("a", "b")
    result: Bundle = sync_row.__op__(DummyContext, bundle, table="rows").output

    assert len(injected) == 2
    assert all(c is DummyContext for c in injected), (
        "Each row must receive the same OpContext instance"
    )
    assert list(result.dfs["rows"]["label"]) == ["a_done", "b_done"]


def test_sync_elementwise_without_opctx_still_works():
    @op(ENV, "Sync no ctx")
    @elementwise(input_table="table")
    def sync_no_ctx(record: Record, *, table: str, suffix: str = "_x") -> None:
        record["label"] = record["label"] + suffix

    bundle = _make_bundle("foo", "bar")
    result: Bundle = sync_no_ctx.__op__(DummyContext, bundle, table="rows", suffix="_z").output

    assert list(result.dfs["rows"]["label"]) == ["foo_z", "bar_z"]


@pytest.mark.asyncio
async def test_async_elementwise_injects_opcontext():
    injected: list[OpContext] = []

    @op(ENV, "Async inject")
    @elementwise(input_table="table")
    async def async_row(record: Record, op_ctx: OpContext, *, table: str) -> None:
        injected.append(op_ctx)
        record["label"] = record["label"] + "_async"

    bundle = _make_bundle("x", "y")
    result: Bundle = await async_row.__op__(DummyContext, bundle, table="rows").output

    assert len(injected) == 2
    assert all(c is DummyContext for c in injected)
    assert list(result.dfs["rows"]["label"]) == ["x_async", "y_async"]


@pytest.mark.asyncio
async def test_async_elementwise_without_opctx_still_works():
    @op(ENV, "Async no ctx")
    @elementwise(input_table="table")
    async def async_no_ctx(record: Record, *, table: str) -> None:
        record["label"] = record["label"] + "_done"

    bundle = _make_bundle("p", "q")
    result: Bundle = await async_no_ctx.__op__(DummyContext, bundle, table="rows").output

    assert list(result.dfs["rows"]["label"]) == ["p_done", "q_done"]
