"""Operations for tables."""

import enum

from lynxkite_core import ops
import pandas as pd
import io
from .. import core

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


@op("Vector from attribute pair", icon="link")
def vector_from_attribute_pair(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute1: core.ColumnNameByTableName,
    attribute2: core.ColumnNameByTableName,
    new_name: str,
) -> core.Bundle:
    """Creates a new column with vectors that contain the two attributes"""
    b = b.copy()
    df = b.dfs[table_name]
    df[new_name] = list(zip(df[attribute1], df[attribute2]))
    return b


@op("Add rank attribute", icon="link")
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


@op("Rename table", color="orange", icon="table-filled")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    return b


@op("Select Table", icon="table-filled")
def select_table(b: core.Bundle, *, table_name: core.TableName) -> core.Bundle:
    df = b.dfs[table_name]
    bundle = core.Bundle()
    bundle.dfs[table_name] = df
    return bundle


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

    df_a = bundle_a.dfs[table_a]
    df_b = bundle_b.dfs[table_b]

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

    return core.Bundle(dfs={"merged": merged_df})
