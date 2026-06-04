"""Elementwise decorator for row-by-row DataFrame operations."""

from __future__ import annotations

import inspect
import asyncio
import typing
import pandas as pd

from lynxkite_core.ops import OpContext

try:
    from lynxkite_enterprise import lim_worker
    from lynxkite_enterprise.lim import (
        LimConfig,
        call_lim_worker,
        ensure_lim_worker_ready,
        is_lim_worker,
        lim_to_k8s_needs_kwargs,
        make_service_name,
    )
    from lynxkite_enterprise import k8s

    enterpise_backend = True
except ImportError:
    enterpise_backend = False

from .bundle import Bundle
from .record import ColumnKey, Record, RowIndex


def elementwise(
    func: typing.Callable | None = None,
    *,
    input_table: str,
    desc: str = "",
    concurrency: int = 1,
    lim_config: typing.Any = None,
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

        if enterpise_backend:
            service_name = make_service_name(func)
            if lim_config and is_lim_worker():
                lim_worker.register_lim_function(
                    service_name,
                    func,
                    route_prefix=lim_config.get("route_prefix", "/lim"),
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
                if enterpise_backend:
                    lim_worker_mode = is_lim_worker()
                    if lim_config and not lim_worker_mode:
                        return await _elementwise_impl_async_remote(
                            self,
                            bundle,
                            selected_input_table,
                            kwargs,
                            desc=desc,
                            concurrency=concurrency,
                            service_name=service_name,
                            lim_config=lim_config,
                        )
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


async def _elementwise_impl_async_remote(
    self: OpContext,
    bundle: Bundle,
    input_table_selection: typing.Any,
    kwargs: dict[str, typing.Any],
    *,
    desc: str,
    concurrency: int,
    service_name: str,
    lim_config: typing.Any,
) -> Bundle:
    bundle, table_name, df = _prepare_elementwise(
        bundle=bundle,
        input_table_selection=input_table_selection,
    )
    needs_kwargs = lim_to_k8s_needs_kwargs(service_name=service_name, lim_config=lim_config)
    route_prefix = str(lim_config.get("route_prefix", "/lim"))
    endpoint_path = f"{route_prefix}/{service_name}".replace("//", "/")
    timeout = max(1, int(lim_config.get("timeout_seconds", 600)))
    effective_concurrency = max(1, int(lim_config.get("allow_concurrent_inputs", concurrency)))

    async def dispatch_rows(op_ctx: OpContext):
        namespace = needs_kwargs.get("namespace", "default")
        ip = k8s.get_ip(service_name, namespace=namespace)
        # k8s Service is exposed on port 80 and forwards to container port.
        port = 80
        endpoint = f"http://{ip}:{port}{endpoint_path}"
        health_endpoint = f"http://{ip}:{port}/lim/health"

        health_payload = await ensure_lim_worker_ready(
            endpoint=health_endpoint, timeout=min(timeout, 30)
        )
        services = health_payload.get("services", [])
        if service_name not in services:
            raise RuntimeError(
                f"LIM worker is healthy but service '{service_name}' is not registered. "
                f"Available services: {services}"
            )

        semaphore = asyncio.Semaphore(effective_concurrency)
        rows = list(enumerate(df.iterrows()))

        async def process_one(row_pos: int, idx: RowIndex, row: pd.Series):
            async with semaphore:
                updates = await call_lim_worker(
                    endpoint=endpoint,
                    payload={
                        "row_keys": row.index.tolist(),
                        "row_values": row.tolist(),
                        "index": idx,
                        "kwargs": kwargs,
                    },
                    timeout=timeout,
                )
                return row_pos, updates

        pending = [process_one(row_pos, idx, row) for row_pos, (idx, row) in rows]
        for completed in op_ctx.tqdm(asyncio.as_completed(pending), total=len(rows), desc=desc):
            row_pos, updates = await completed
            _apply_updates(df, row_pos, updates)

    decorated_dispatch = k8s.needs(name=service_name, **needs_kwargs)(dispatch_rows)
    await decorated_dispatch(self)

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
