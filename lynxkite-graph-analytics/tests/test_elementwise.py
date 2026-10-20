"""Regression tests for table behavior in @elementwise row functions."""

from __future__ import annotations
import pandas as pd
import pytest

from lynxkite_core.opcontext import OpContext
from lynxkite_core.ops import op
from lynxkite_graph_analytics.core import TableColumn
from lynxkite_graph_analytics.bundle import Bundle
from lynxkite_graph_analytics.elementwise import elementwise
from lynxkite_graph_analytics.record import Record

ENV = "test_elementwise_opctx"
DummyContext = OpContext(
    op=None
)  # For testing the op decorator without needing a full execution context.


def test_records_not_first_is_rejected():
    """@elementwise row function must have Record as first positional parameter."""

    with pytest.raises(ValueError, match="'record: Record' as first parameter"):

        @op(ENV, "Bad record position")
        @elementwise(input_table="table")
        def bad_record_pos(op_ctx: OpContext, record: Record, *, table: str) -> None:
            pass


def test_dynamic_input_table_sync_reads_correct_df():
    """Sync elementwise with table name reads from that DataFrame."""

    @op(ENV, "Dynamic sync")
    @elementwise(input_table="table")
    def tag_dyn(
        record: Record, *, table: TableColumn = ("proteins", "label"), suffix: str = "_x"
    ) -> None:
        record["label"] = record["label"] + suffix

    df = pd.DataFrame({"label": ["a", "b"]})
    bundle = Bundle(dfs={"proteins": df})
    result: Bundle = tag_dyn.__op__(
        DummyContext, bundle, table=("proteins", "label"), suffix="_z"
    ).output

    assert list(result.dfs["proteins"]["label"]) == ["a_z", "b_z"]


@pytest.mark.asyncio
async def test_dynamic_input_table_async_reads_correct_df():
    """Async elementwise with a dynamic table name reads from that DataFrame."""

    @op(ENV, "Dynamic async")
    @elementwise(input_table="table")
    async def tag_dyn_async(record: Record, *, table: str = "proteins", suffix: str = "_x") -> None:
        record["label"] = record["label"] + suffix

    df = pd.DataFrame({"label": ["x", "y"]})
    bundle = Bundle(dfs={"proteins": df})
    result: Bundle = await tag_dyn_async.__op__(
        DummyContext, bundle, table="proteins", suffix="_w"
    ).output

    assert list(result.dfs["proteins"]["label"]) == ["x_w", "y_w"]
