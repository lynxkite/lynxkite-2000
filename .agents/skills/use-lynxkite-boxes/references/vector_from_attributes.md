**Vector from attributes:**
Creates a new column with vectors that contain the selected attributes in the selected order
```python
@op("Vector from attributes", icon="brackets-contain")
def vector_from_attributes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attributes: core.MultiColumnNameByTableName,
    vector_name: str,
) -> core.Bundle:
    """Creates a new column with vectors that contain the selected attributes in the selected order"""
    b = b.copy()
    df = b.dfs[table_name].copy()
    df[vector_name] = list(zip(*(df[col] for col in attributes)))
    b.dfs[table_name] = df
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - attributes: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
