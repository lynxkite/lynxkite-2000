"""Operations for tables."""

import enum

from lynxkite_core import ops
import pandas as pd
import io
from .. import core, bundle

op = ops.op_registration(core.ENV, "Table operations")


@op("Sample table", icon="filter-filled")
def sample_table(b: core.Bundle, *, table_name: core.TableName = "meta", fraction: float = 0.1):
    b = b.copy()
    b.dfs[table_name] = b.dfs[table_name].sample(frac=fraction)
    return b


@op("Filter with formula", icon="filter-filled")
def filter_with_formula(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    """Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false"""
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.query(formula)
    return b


class OrderType(enum.StrEnum):
    asc = "ascending"
    desc = "descending"


@op("Vector from attributes", icon="brackets-contain")
def vector_from_attributes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attributes: core.MultiColumnNameByTableName,
    vector_name: str,
) -> core.Bundle:
    """Creates a new column with vectors that contain the selected attributes in the selected order"""
    b = b.copy()
    df = b.dfs[table_name].copy()
    df[vector_name] = list(zip(*(df[col] for col in attributes)))
    b.dfs[table_name] = df
    return b


@op("Add rank attribute", icon="sort-descending")
def add_rank(
    b: core.Bundle,
    *,
    table_column: core.TableColumn,
    rank_name: str,
    order: OrderType,
):
    """Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row

    Parameters
    ----------
    table_column : core.TableColumn
        The table and column to rank
    rank_name : str
        The name of the new rank column
    order : OrderType
        The order in which to rank the rows either 'ascending' or 'descending'

    Returns
    -------
    output : core.Bundle
        The updated bundle with the new rank column
    """
    table, column = table_column
    b = b.copy()
    df = b.dfs[table]

    df = df.sort_values(by=column, ascending=(order == OrderType.asc))
    df[rank_name] = range(len(df))

    b.dfs[table] = df
    return b


@op("Filter tables", color="orange", icon="table-minus")
def filter_tables(
    b: core.Bundle, *, drop_selected: bool, tables: core.MultiTableName
) -> core.Bundle:
    """
    Keeps/removes the selected tables based on the value of drop_selected
    :param b: the bundle
    :param drop_selected: if True, removes the selected tables, otherwise keeps them
    :param tables: the tables to keep/remove
    """
    b = b.copy()

    b.dfs = {k: v for k, v in b.dfs.items() if (k in tables) != drop_selected}
    b.relations = [r for r in b.relations if r.source_table in b.dfs and r.target_table in b.dfs]
    return b


@op("Fill attributes with default values", icon="table-column")
def fill_with_default(
    b: core.Bundle, *, table_name: core.TableName, adder: core.DropdownTextAdderByTableName
) -> core.Bundle:
    """
    An attribute may not be defined everywhere. This operation sets the provided values for the rows of the specified attributes where they are not defined.
    :param b: the bundle
    :param table_name: the table to operate on
    :param adder: the attributes and the values to set
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    for column, default_value in adder:
        df[column] = df[column].fillna(default_value)
    b.dfs[table_name] = df
    return b


@op("Rename table", color="orange", icon="writing")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    relations = []
    for r in b.relations:
        r = r.copy()
        if r.source_table == old_name:
            r.source_table = new_name
        if r.target_table == old_name:
            r.target_table = new_name
        relations.append(r)
    b.relations = relations
    return b


@op("Derive property", icon="arrow-big-right-lines")
def derive_property(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.eval(formula)
    return b


@op("Enter table data", color="green", icon="table-filled")
def enter_table_data(
    *,
    table_name: str,
    data: ops.LongStr,
):
    """Enter table data as CSV. The first row should contain column names."""
    b = core.Bundle()
    b.dfs[table_name] = pd.read_csv(io.StringIO(data))
    return b


class JoinType(enum.StrEnum):
    inner = "inner"
    outer = "outer"
    left = "left"
    right = "right"
    cross = "cross"


@op("Join tables", color="orange", icon="link")
def join_tables(
    bundle_a: core.Bundle,
    bundle_b: core.Bundle,
    *,
    table_a: core.TableName,
    table_b: core.TableName,
    join_type: JoinType = JoinType.inner,
    on_column: str = "",
    left_on: str = "",
    right_on: str = "",
    suffixes: str = "_a,_b",
):
    """
    Join/merge dataframes from two bundles.

    Parameters:
    - table_a: Table name from bundle A
    - table_b: Table name from bundle B
    - join_type: Type of join - "inner", "outer", "left", "right", "cross"
    - on_column: Column name to join on (same name in both tables)
    - left_on: Column name in left table (when column names differ)
    - right_on: Column name in right table (when column names differ)
    - suffixes: Suffixes for overlapping columns (comma-separated, e.g., "_a,_b")
    """
    bundle_a = bundle_a.copy()
    bundle_b = bundle_b.copy()
    df_a = bundle_a.dfs[table_a].copy()
    df_b = bundle_b.dfs[table_b].copy()

    # Parse suffixes
    suffix_parts = [s.strip() for s in suffixes.split(",")]
    if len(suffix_parts) != 2:
        suffix_list: tuple[str, str] = ("_a", "_b")
    else:
        suffix_list = (suffix_parts[0], suffix_parts[1])

    # Perform the join
    if on_column:
        merged_df = pd.merge(df_a, df_b, on=on_column, how=join_type.value, suffixes=suffix_list)
    elif left_on and right_on:
        merged_df = pd.merge(
            df_a,
            df_b,
            left_on=left_on,
            right_on=right_on,
            how=join_type.value,
            suffixes=suffix_list,
        )
    else:
        # Join on index if no columns specified
        merged_df = pd.merge(
            df_a, df_b, left_index=True, right_index=True, how=join_type.value, suffixes=suffix_list
        )
    b = bundle.merge_bundles([bundle_a, bundle_b], merge_mode=bundle.BundleMergeMode.must_be_unique)

    new_table = "merged"
    table_suffixes = {table_a: suffix_list[0], table_b: suffix_list[1]}

    for r in b.relations:
        if r.source_table in table_suffixes:
            suffixed_key = r.source_key + table_suffixes[r.source_table]
            r.source_table = new_table
            if suffixed_key in merged_df.columns:
                r.source_key = suffixed_key

        if r.target_table in table_suffixes:
            suffixed_key = r.target_key + table_suffixes[r.target_table]
            r.target_table = new_table
            if suffixed_key in merged_df.columns:
                r.target_key = suffixed_key

    b.dfs.pop(table_a, None)
    b.dfs.pop(table_b, None)
    b.dfs[new_table] = merged_df
    return b
