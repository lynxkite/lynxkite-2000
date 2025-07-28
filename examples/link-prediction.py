from lynxkite_core import ops
from lynxkite_graph_analytics import core


op = ops.op_registration("LynxKite Graph Analytics")


@op("Add incrementing ID")
def add_incrementing_id(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    id_column: core.ColumnNameByTableName,
    save_as: str = "result",
):
    """Add an incrementing ID column starting from 0 to a table."""
    bundle = bundle.copy()
    df = bundle.dfs[table_name].copy()
    df[id_column] = range(len(df))
    bundle.dfs[save_as] = df
    return bundle


@op("Randomly sample table")
def random_sample(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    ratio: float,
    seed: int,
):
    bundle = bundle.copy()
    sampled = bundle.dfs[table_name].sample(frac=ratio, random_state=seed)
    bundle.dfs[f"{table_name}_sampled"] = sampled
    return bundle
