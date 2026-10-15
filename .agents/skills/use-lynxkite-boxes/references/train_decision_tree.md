**Train decision tree regression model:**

```python
@op("Train decision tree regression model", icon="circles")
def train_decision_tree(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    feature_column: core.ColumnNameByTableName,
    label_column: core.ColumnNameByTableName,
    max_depth: int = 7,
    min_impurity_decrease: float = 0.0,
    min_samples_leaf: int = 1,
    seed: int = 42,
    model_name: str = "decision_tree",
) -> core.Bundle:
    """
    :param seed: seed for random number generator.
    :param min_samples_leaf: minimum number of samples required to be at a leaf node.
    :param min_impurity_decrease: minimum impurity decrease required to split a node.
    :param max_depth: maximum depth of the tree.
    :param b: The bundle.
    :param table_name: The name of the table containing the training data.
    :param feature_column: The name of the column containing the feature vectors.
    :param label_column: The name of the column containing the labels.
    :param model_name: The name to assign to the trained model.
    """
    b = b.copy()
    train_df = b.dfs[table_name].copy()
    x_train = np.array(train_df[feature_column].tolist())
    y_train = train_df[label_column].to_numpy()
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_impurity_decrease=min_impurity_decrease,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    tree = tree.fit(x_train, y_train)
    b.other[model_name] = tree
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - feature_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
