**One-hot encoding:**
Creates a new table with the one-hot encoded versions of the specified columns.
```python
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

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - columns: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
