"""Basic operations for this environment."""

from lynxkite_core import ops

from .. import core
import enum
import io
import json
import pandas as pd


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
def organize(bundles: list[core.Bundle], *, relations: str = ""):
    """Merge multiple inputs and construct graphs from the tables.

    To create a graph, import tables for edges and nodes, and combine them in this operation.
    """
    bundle = core.Bundle()
    for b in bundles:
        bundle.dfs.update(b.dfs)
        bundle.relations.extend(b.relations)
        bundle.other.update(b.other)
    if relations.strip():
        bundle.relations = [core.RelationDefinition(**r) for r in json.loads(relations).values()]
    return ops.Result(output=bundle, display=bundle.to_table_view(limit=100))


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
