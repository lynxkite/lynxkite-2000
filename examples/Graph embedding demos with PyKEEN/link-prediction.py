from lynxkite_core import ops
from lynxkite_graph_analytics import core

import numpy as np


op = ops.op_registration("LynxKite Graph Analytics")


@op("Randomly sample table", color="orange", icon="filter-filled")
def random_sample(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    ratio: float,
    seed: int,
):
    """
    This WILL overwrite your original table.
    """
    bundle = bundle.copy()
    sampled = bundle.dfs[table_name].sample(frac=ratio, random_state=seed)
    bundle.dfs[table_name] = sampled
    return bundle


@op("Split temporal data into test/validation", color="orange", icon="circle-half-2")
def split_temporal_data(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    test_ratio: float,
    seed: int,
):
    bundle = bundle.copy()
    df = bundle.dfs[table_name].copy()
    test, val = np.split(df.sample(frac=1, random_state=seed), [int((1 - test_ratio) * len(df))])
    bundle.dfs[f"{table_name}_test"] = test
    bundle.dfs[f"{table_name}_val"] = val
    return bundle
