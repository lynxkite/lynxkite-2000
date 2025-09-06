from lynxkite_core import ops
from lynxkite_graph_analytics import core


op = ops.op_registration("LynxKite Graph Analytics")


@op("Combine data")
def combine_data(
    bundle1: core.Bundle,
    bundle2: core.Bundle,
):
    combined_bundle = core.Bundle()
    combined_bundle.dfs = {**bundle1.dfs, **bundle2.dfs}
    return combined_bundle


@op("Train/test/validation split")
def train_test_validation_split(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    bundle = bundle.copy()
    df_edges = bundle.dfs[table_name]
    df_train = df_edges.sample(frac=train_ratio, random_state=seed)
    df_test = df_edges.drop(df_train.index).sample(
        frac=test_ratio / (test_ratio + val_ratio), random_state=seed
    )
    df_val = df_edges.drop(df_train.index).drop(df_test.index)

    bundle.dfs[f"{table_name}_train"] = df_train
    bundle.dfs[f"{table_name}_test"] = df_test
    bundle.dfs[f"{table_name}_val"] = df_val

    return bundle
