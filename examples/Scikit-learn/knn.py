"""Custom operations for the KNN demo workspace."""

from collections.abc import Iterable

import pandas as pd

from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core
from sklearn.datasets import fetch_openml

op = op_registration(core.ENV, "KNN")


@op("Fetch openml dataset")
def fetch_dataset(*, dataset_name: str, version: int = 1) -> core.Bundle:
    b = core.Bundle()
    dataset = fetch_openml(dataset_name, version=version, as_frame=True)
    b.dfs[dataset_name] = dataset.frame
    return b


@op("One-hot encoding")
def one_hot(
    b: core.Bundle, *, table_name: core.TableName, columns: core.MultiColumnNameByTableName
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name].copy()

    for col in columns:
        dummies = pd.get_dummies(df[col])
        df[col] = list(map(tuple, dummies.to_numpy()))

    b.dfs[table_name + "_one_hot"] = df
    return b


def _recursive_flatten(item):
    flat = []
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        for i in item:
            flat.extend(_recursive_flatten(i))
    else:
        flat.append(item)
    return flat


@op("Flatten column")
def flatten_column(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    column_name: str,
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name].copy()

    df[column_name] = df[column_name].apply(lambda x: tuple(_recursive_flatten(x)))
    b.dfs[table_name] = df
    return b
