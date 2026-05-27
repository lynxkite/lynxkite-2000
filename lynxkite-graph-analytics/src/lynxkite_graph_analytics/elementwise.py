"""Elementwise decorator for row-by-row DataFrame operations."""

from __future__ import annotations

import inspect
import asyncio
import typing
import pandas as pd

from lynxkite_core.ops import OpContext
from .bundle import Bundle
from .record import ColumnKey, Record, RowIndex


def elementwise(
    func: typing.Callable | None = None,
    *,
    input_table: str,
    desc: str = "",
    concurrency: int = 1,
) -> typing.Callable:
    """Decorator for operations that process each row independently.

    The wrapped function receives a Record and can update output fields directly.
    Iteration, progress display, and DataFrame writes are handled by the decorator.

    Example:
        @elementwise(input_table="protein_table_column", desc="Processing proteins")
        async def process(record: Record, *, protein_table_column: tuple[str, str]):
            record.result = await api_call(record[protein_table_column])
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params or params[0].name != "record":
            raise ValueError(
                f"@elementwise requires 'record: Record' as first parameter, got: {params[0].name if params else 'none'}"
            )

        public_signature = sig.replace(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("bundle", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *[p for p in params[1:] if p.name != "bundle"],
            ]
        )

        if inspect.iscoroutinefunction(func):

            async def async_wrapper(
                self: OpContext,
                bundle: Bundle,
                **kwargs,
            ) -> Bundle:
                selected_input_table = kwargs[input_table]
                return await _elementwise_impl_async(
                    self,
                    bundle,
                    func,
                    selected_input_table,
                    kwargs,
                    desc=desc,
                    concurrency=concurrency,
                )

            setattr(async_wrapper, "__signature__", public_signature)
            return async_wrapper
        else:

            def sync_wrapper(self: OpContext, bundle: Bundle, **kwargs) -> Bundle:
                selected_input_table = kwargs[input_table]
                return _elementwise_impl_sync(
                    self,
                    bundle,
                    func,
                    selected_input_table,
                    kwargs,
                    desc=desc,
                )

            setattr(sync_wrapper, "__signature__", public_signature)
            return sync_wrapper

    return decorator if func is None else decorator(func)


async def _elementwise_impl_async(
    self: OpContext,
    bundle: Bundle,
    func: typing.Callable,
    input_table_selection: typing.Any,
    kwargs: dict[str, typing.Any],
    *,
    desc: str,
    concurrency: int,
) -> Bundle:
    bundle, table_name, df = _prepare_elementwise(
        bundle=bundle,
        input_table_selection=input_table_selection,
    )

    concurrency = max(1, int(concurrency))

    if concurrency == 1:
        for row_pos, (idx, row) in self.tqdm(enumerate(df.iterrows()), total=len(df), desc=desc):
            record = Record(row, idx)
            await func(record, **kwargs)
            _apply_record_updates(df, row_pos, record)
    else:
        semaphore = asyncio.Semaphore(concurrency)
        rows = list(enumerate(df.iterrows()))

        async def process_one(row_pos: int, idx: RowIndex, row: pd.Series):
            async with semaphore:
                record = Record(row, idx)
                await func(record, **kwargs)
                return row_pos, record.get_updates()

        pending = [process_one(row_pos, idx, row) for row_pos, (idx, row) in rows]
        for completed in self.tqdm(asyncio.as_completed(pending), total=len(rows), desc=desc):
            row_pos, updates = await completed
            _apply_updates(df, row_pos, updates)

    bundle.dfs[table_name] = df
    return bundle


def _elementwise_impl_sync(
    self: OpContext,
    bundle: Bundle,
    func: typing.Callable,
    input_table_selection: typing.Any,
    kwargs: dict[str, typing.Any],
    *,
    desc: str,
) -> Bundle:
    bundle, table_name, df = _prepare_elementwise(
        bundle=bundle,
        input_table_selection=input_table_selection,
    )

    for row_pos, (idx, row) in self.tqdm(enumerate(df.iterrows()), total=len(df), desc=desc):
        record = Record(row, idx)
        func(record, **kwargs)
        _apply_record_updates(df, row_pos, record)

    bundle.dfs[table_name] = df
    return bundle


def _prepare_elementwise(
    *,
    bundle: Bundle,
    input_table_selection: typing.Any,
) -> tuple[Bundle, str, pd.DataFrame]:
    table_name = _resolve_input_table_name(input_table_selection)
    bundle = bundle.copy()
    df = bundle.dfs[table_name].copy()

    return bundle, table_name, df


def _resolve_input_table_name(input_table_selection: typing.Any) -> str:
    if isinstance(input_table_selection, str):
        return input_table_selection

    if isinstance(input_table_selection, (tuple, list)):
        if len(input_table_selection) != 2:
            raise ValueError("Input table selector pair must be (table_name, column_name)")
        table_name, _ = input_table_selection
        return table_name

    raise TypeError(
        "Unsupported input table selector type. Expected table name string or (table, column) pair."
    )


def _apply_record_updates(df: pd.DataFrame, row_pos: int, record: Record) -> None:
    _apply_updates(df, row_pos, record.get_updates())


def _apply_updates(
    df: pd.DataFrame,
    row_pos: int,
    updates: dict[ColumnKey, typing.Any],
) -> None:
    for col, value in updates.items():
        if col not in df.columns:
            df[col] = pd.NA
        col_pos = list(df.columns).index(col)
        df.iat[row_pos, col_pos] = value
