"""Basic operations for this environment."""

from lynxkite_core import ops

from .. import core
import json
import io
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


@op("Enter table data", icon="table-filled")
def enter_table_data(
    *,
    table_name: str,
    data: ops.LongStr,
):
    """Enter table data as CSV. The first row should contain column names."""
    b = core.Bundle()
    b.dfs[table_name] = pd.read_csv(io.StringIO(data))
    return b
