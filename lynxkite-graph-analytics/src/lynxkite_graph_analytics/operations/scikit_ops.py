"""Operations related to the scikit-learn library."""

import enum
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from .. import core
from lynxkite_core.ops import op_registration
from sklearn.datasets import fetch_openml

op = op_registration(core.ENV, "Scikit")


@op("Fetch OpenML dataset", slow=True, color="blue", icon="database-import")
def fetch_dataset(*, dataset_name: str, version: int = 1) -> core.Bundle:
    """
    Fetches the specified dataset from OpenML.
    :param dataset_name: The name of the dataset to fetch.
    :param version: The version of the dataset to fetch.
    """
    b = core.Bundle()
    dataset = fetch_openml(dataset_name, version=version, as_frame=True)
    b.dfs[dataset_name] = dataset.frame
    return b


@op("One-hot encoding", icon="binary")
def one_hot(
    b: core.Bundle, *, table_name: core.TableName, columns: core.MultiColumnNameByTableName
) -> core.Bundle:
    """
    Creates a new table with the one-hot encoded versions of the specified columns.
    :param b: The bundle.
    :param table_name: The name of the table.
    :param columns: The columns to one-hot encode.
    """
    b = b.copy()
    df = b.dfs[table_name].copy()

    for col in columns:
        dummies = pd.get_dummies(df[col])
        df[col + "_one_hot"] = list(map(tuple, dummies.to_numpy()))

    b.dfs[table_name] = df
    return b


def _recursive_flatten(item):
    flat = []
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        for i in item:
            flat.extend(_recursive_flatten(i))
    else:
        flat.append(item)
    return flat


@op("Flatten column", icon="ironing")
def flatten_column(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    column_name: core.ColumnNameByTableName,
) -> core.Bundle:
    """
    Flattens the items in the specified column of the specified table.
    :param b: The bundle
    :param table_name:  the name of the table
    :param column_name:  the name of the column whose items should be flattened
    """
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


@op("Train K-nearest neighbors classifier", icon="circles")
def train_knn(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    n_neighbors: int = 5,
    model_name: str = "knn",
    weights: Literal["uniform", "distance"] = "uniform",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    leaf_size: int = 30,
    metric: DistanceMetric = DistanceMetric.minkowski,
    p: int | None = 2,
    n_jobs: int | None = None,
) -> core.Bundle:
    """
    Trains a K-nearest neighbors classifier on the data from the specified table.
    :param b: The bundle.
    :param table_name: The name of the table containing the training data.
    :param feature_column: The name of the column containing the feature vectors.
    :param label_column: The name of the column containing the labels.
    :param n_neighbors: The number of neighbors to use.
    :param model_name: The name to assign to the trained model.
    :param weights: The weight function used in prediction.
    :param algorithm: The algorithm used to compute the nearest neighbors.
    :param leaf_size: The leaf size passed to the algorithm.
    :param metric: The distance metric to use.
    :param p: The power parameter for the Minkowski metric.
    :param n_jobs: The number of parallel jobs to run.
    """
    b = b.copy()
    train_df = b.dfs[table_name].copy()
    x_train = np.array(train_df[feature_column].tolist())
    y_train = train_df[label_column].to_numpy()

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric.value,
        n_jobs=n_jobs,
    )
    knn = knn.fit(x_train, y_train)
    b.other[model_name] = knn
    return b


@op("Make prediction", icon="circles")
def scikit_predict(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    prediction_column: str,
    model_name: str,
) -> core.Bundle:
    """
    Makes predictions using a trained scikit-learn model on the data from the specified table.
    :param b: The bundle.
    :param table_name: The name of the table containing the data.
    :param feature_column: The name of the column containing the feature vectors.
    :param prediction_column: The name of the new column with the predictions.
    :param model_name: The name of the trained model to use for the predictions.
    """
    b = b.copy()
    test_df = b.dfs[table_name].copy()
    x_test = np.array(test_df[feature_column].tolist())

    model = b.other[model_name]
    test_df[prediction_column] = model.predict(x_test)

    b.dfs[table_name] = test_df
    return b


@op("Confusion matrix", view="matplotlib", color="blue", icon="dots-diagonal-2")
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
    fig, ax = plt.subplots(tight_layout=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xlabel("Predicted", labelpad=15)
    plt.ylabel("Actual", labelpad=15)
    plt.yticks(rotation=0)
