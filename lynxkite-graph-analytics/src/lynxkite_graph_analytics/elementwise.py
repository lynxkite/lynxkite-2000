"""Elementwise decorator for row-by-row DataFrame operations."""

from __future__ import annotations

import asyncio
import inspect
import typing
import pandas as pd

from lynxkite_core.ops import OpContext

try:
    from lynxkite_enterprise.execution import execution_parallelism  # ty: ignore[unresolved-import]
    from lynxkite_enterprise.lim import (  # ty: ignore[unresolved-import]
        get_remote_row_processor,
        register_lim_function_if_worker,
    )

    enterprise_backend = True
except ImportError:
    enterprise_backend = False
    execution_parallelism = None  # type: ignore[assignment]

from .bundle import Bundle
from .record import ColumnKey, Record, RowIndex


def elementwise(
    func: typing.Callable | None = None,
    *,
    input_table: str,
    desc: str = "",
    lim_config: typing.Any = None,
) -> typing.Callable:
    """Decorator for operations that process each row independently.

    The wrapped function receives a Record and can update output fields directly.
    Iteration, progress display, and DataFrame writes are handled by the decorator.

    Example:
        @elementwise(input_table="protein_table_column", desc="Processing proteins")
        async def process(record: Record, *, protein_table_column: TableColumn):
            record.result = await api_call(record[protein_table_column])

    With LIM, pass a small ``lim_config``; storage and k8s defaults are filled in
    automatically::

        @elementwise(
            input_table="protein_table_column",
            desc="ESM2",
            lim_config={"cpu": "4", "memory": "8Gi"},
        )
        async def query_esm2(record: Record, *, protein_table_column: TableColumn):
            record.embeddings = my_inference(record[protein_table_column])
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params or params[0].name != "record":
            raise ValueError(
                f"@elementwise requires 'record: Record' as first parameter, got: {params[0].name if params else 'none'}"
            )

        service_name = None
        if enterprise_backend and lim_config:
            service_name = register_lim_function_if_worker(func, lim_config)

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
                    lim_config=lim_config,
                    service_name=service_name,
                )

            setattr(async_wrapper, "__signature__", public_signature)
            return async_wrapper
        else:
            if lim_config:
                raise ValueError("@elementwise(lim_config=...) supports async functions only")

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
    lim_config: typing.Any | None,
    service_name: str | None,
) -> Bundle:
    bundle, table_name, df = _prepare_elementwise(
        bundle=bundle,
        input_table_selection=input_table_selection,
    )

    async def process_row_updates(idx: RowIndex, row: pd.Series) -> dict[ColumnKey, typing.Any]:
        record = Record(row, idx)
        await func(record, **kwargs)
        return record.get_updates()

    lim_cleanup = None
    if service_name and lim_config:
        remote_processor = await get_remote_row_processor(
            self,
            kwargs,
            service_name=service_name,
            lim_config=lim_config,
        )
        if remote_processor is not None:
            process_row_updates = remote_processor  # type: ignore[assignment]
            lim_cleanup = getattr(remote_processor, "lim_cleanup", None)

    try:
        await _process_rows_concurrently(
            self,
            df,
            process_row_updates,
            desc=desc,
            parallelism=execution_parallelism(self) if execution_parallelism else 1,
        )
    finally:
        if lim_cleanup is not None:
            await lim_cleanup()

    bundle.dfs[table_name] = df
    return bundle


async def _process_rows_concurrently(
    op_ctx: OpContext,
    df: pd.DataFrame,
    process_row_updates: typing.Callable[
        [RowIndex, pd.Series], typing.Awaitable[dict[ColumnKey, typing.Any]]
    ],
    *,
    desc: str,
    parallelism: int,
) -> None:
    rows = list(enumerate(df.iterrows()))
    if not rows:
        return

    semaphore = asyncio.Semaphore(parallelism)

    async def process_one_row(
        row_pos: int, idx: RowIndex, row: pd.Series
    ) -> tuple[int, dict[ColumnKey, typing.Any]]:
        async with semaphore:
            updates = await process_row_updates(idx, row)
        return row_pos, updates

    tasks = [
        asyncio.create_task(process_one_row(row_pos, idx, row)) for row_pos, (idx, row) in rows
    ]
    try:
        for done in op_ctx.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            row_pos, updates = await done
            _apply_updates(df, row_pos, updates)
    except BaseException:
        for task in tasks:
            task.cancel()
        raise


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
        _apply_updates(df, row_pos, record.get_updates())

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
