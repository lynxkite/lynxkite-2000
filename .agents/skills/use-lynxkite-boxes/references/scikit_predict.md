**Make prediction:**
Makes predictions using a trained scikit-learn model on the data from the specified table.
```python
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

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - feature_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
