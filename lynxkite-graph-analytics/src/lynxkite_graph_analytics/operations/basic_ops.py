"""Basic operations for this environment."""

from lynxkite_core import ops

from .. import core, bundle
import enum
import io
import json
import pandas as pd
import numpy as np


op = ops.op_registration(core.ENV)


@op("Sample table", icon="filter-filled")
def sample_table(b: core.Bundle, *, table_name: core.TableName = "meta", fraction: float = 0.1):
    b = b.copy()
    b.dfs[table_name] = b.dfs[table_name].sample(frac=fraction)
    return b


@op("View tables", view="table_view", color="blue", icon="table-filled")
def view_tables(bundle: core.Bundle, *, _tables_open: str = "", limit: int = 100):
    _tables_open = _tables_open  # The frontend uses this parameter to track which tables are open.
    return bundle.to_table_view(limit=limit)


@op(
    "Organize",
    view="graph_creation_view",
    outputs=["output"],
    icon="settings-filled",
)
def organize(
    bundles: list[core.Bundle],
    *,
    relations: str = "",
    merge_mode: bundle.BundleMergeMode = bundle.BundleMergeMode.must_be_unique,
):
    """Merge multiple inputs and construct graphs from the tables.

    To create a graph, import tables for edges and nodes, and combine them in this operation.
    """
    b = bundle.merge_bundles(bundles, merge_mode=merge_mode)
    if relations.strip():
        b.relations = [core.RelationDefinition(**r) for r in json.loads(relations).values()]
    return ops.Result(output=b, display=b.to_table_view(limit=100))


@op("Derive property", icon="arrow-big-right-lines")
def derive_property(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.eval(formula)
    return b


@op("Filter with formula")
def filter_with_formula(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    """Removes all rows where the formula evaluates to false"""
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.query(formula)
    return b


@op("Vector from attribute pair", color="orange", icon="link")
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


@op("Rename table", color="orange", icon="link")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    return b


class OrderType(enum.StrEnum):
    asc = "ascending"
    desc = "descending"


@op("Add rank attribute", color="orange", icon="link")
def add_rank(
    b: core.Bundle,
    *,
    table_column: core.TableColumn,
    rank_name: str,
    order: OrderType,
):
    """Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row"""
    table, column = table_column
    b = b.copy()
    df = b.dfs[table]

    df = df.sort_values(by=column, ascending=(order == OrderType.asc))
    df[rank_name] = range(len(df))

    b.dfs[table] = df
    return b


@op("Connect nodes on attribute")
def connect_nodes(
    b: core.Bundle,
    *,
    table1: core.TableName,
    id1: str,
    attribute1: str,
    table2: core.TableName,
    id2: str,
    attribute2: str,
) -> core.Bundle:
    """
    Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

    Parameters:
    - table1: Name of the first table
    - table2: Name of the second table
    - id1: ID column in the first table
    - attribute1: Attribute column in the first table used for matching
    - id2: ID column in the second table
    - attribute2: Attribute column in the second table used for matching
    """
    b = b.copy()

    df1 = (b.dfs[table1]).add_suffix("_src")
    df2 = (b.dfs[table2]).add_suffix("_dst")
    source_key, target_key = id1, id2
    attribute1, attribute2, id1, id2 = (
        attribute1 + "_src",
        attribute2 + "_dst",
        id1 + "_src",
        id2 + "_dst",
    )
    edges = pd.merge(df1, df2, left_on=attribute1, right_on=attribute2)

    if table1 == table2:
        edges[[id1, id2]] = np.sort(edges[[id1, id2]], axis=1)
        edges = edges[edges[id1] != edges[id2]].drop_duplicates()

    b.dfs["edges"] = edges

    b.relations.append(
        core.RelationDefinition(
            name="graph",
            df="edges",
            source_column=id1,
            source_table=table1,
            source_key=source_key,
            target_column=id2,
            target_table=table2,
            target_key=target_key,
        )
    )
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
