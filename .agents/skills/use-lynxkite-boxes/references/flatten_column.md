**Flatten column:**
Flattens the items to 1 dimension in the specified column of the specified table.

If one of the items is the following list: [[a,b],[[c,d],e]]

the flattened version will be: [a,b,c,d,e]
```python
@op("Flatten column", icon="ironing")
def flatten_column(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    column_name: core.ColumnNameByTableName,
) -> core.Bundle:
    """
    Flattens the items to 1 dimension in the specified column of the specified table.

    If one of the items is the following list: [[a,b],[[c,d],e]]

    the flattened version will be: [a,b,c,d,e]
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
