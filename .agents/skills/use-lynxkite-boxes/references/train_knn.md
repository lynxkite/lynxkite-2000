**Train K-nearest neighbors classifier:**
Trains a K-nearest neighbors classifier on the data from the specified table.
```python
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

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - feature_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - weights: typing.Literal['uniform', 'distance']
  - algorithm: typing.Literal['auto', 'ball_tree', 'kd_tree', 'brute']
