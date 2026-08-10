"""Custom operations for the KNN demo workspace."""

import enum
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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


class DistanceMetric(enum.StrEnum):
    minkowski = "minkowski"
    haversine = "haversine"
    russellrao = "russellrao"
    dice = "dice"
    braycurtis = "braycurtis"
    cosine = "cosine"
    canberra = "canberra"
    yule = "yule"
    cityblock = "cityblock"
    sokalsneath = "sokalsneath"
    matching = "matching"
    euclidean = "euclidean"
    l1 = "l1"
    sqeuclidean = "sqeuclidean"
    correlation = "correlation"
    hamming = "hamming"
    chebyshev = "chebyshev"
    rogerstanimoto = "rogerstanimoto"
    jaccard = "jaccard"
    l2 = "l2"
    manhattan = "manhattan"


@op("K-nearest neighbors classifier", icon="circles")
def knn_classifier(
    b: core.Bundle,
    *,
    train_table: core.TableName,
    test_table: core.TableName,
    feature_column: str,
    label_column: str,
    prediction_column: str,
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    leaf_size: int = 30,
    metric: DistanceMetric = DistanceMetric.minkowski,
    p: int | None = 2,
    n_jobs: int | None = None,
) -> core.Bundle:
    b = b.copy()
    train_df = b.dfs[train_table].copy()
    test_df = b.dfs[test_table].copy()

    x_train = np.array(train_df[feature_column].tolist())
    y_train = train_df[label_column].to_numpy()
    x_test = np.array(test_df[feature_column].tolist())

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric.value,
        n_jobs=n_jobs,
    )
    knn.fit(x_train, y_train)
    test_df[prediction_column] = knn.predict(x_test)

    b.dfs[test_table] = test_df
    return b


@op("Confusion matrix", view="visualization", color="blue")
def conf_matrix(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    label_column: core.ColumnNameByTableName,
    prediction_column: core.ColumnNameByTableName,
):
    df = b.dfs[table_name]
    y_true = df[label_column].astype(str)
    y_pred = df[prediction_column].astype(str)
    classes = list(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    max_val = max(int(cm.max()), 1)
    data = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            item = {
                "value": [j, len(classes) - 1 - i, int(val)],
                "itemStyle": {"color": f"rgba(0, 100, 200, {0.1 + 0.9 * (val / max_val)})"},
            }
            data.append(item)

    return {
        "xAxis": {
            "position": "top",
            "data": classes,
            "name": "Predicted",
            "nameLocation": "middle",
            "nameGap": 50,
        },
        "yAxis": {
            "data": classes[::-1],
            "name": "Actual",
            "nameLocation": "middle",
            "nameGap": 60,
        },
        "visualMap": {"show": False},
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": True, "color": "#000000", "fontWeight": "bold"},
            }
        ],
    }
