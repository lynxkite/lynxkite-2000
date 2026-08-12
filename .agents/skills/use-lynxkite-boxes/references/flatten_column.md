**Flatten column:**
Flattens the items in the specified column of the specified table.
```python
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

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - column_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
