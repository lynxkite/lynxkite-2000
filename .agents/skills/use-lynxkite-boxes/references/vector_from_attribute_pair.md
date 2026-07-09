**Vector from attribute pair:**
Creates a new column with vectors that contain the two attributes
```python
@op("Vector from attribute pair", icon="link")
def vector_from_attribute_pair(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute1: core.ColumnNameByTableName,
    attribute2: core.ColumnNameByTableName,
    new_name: str,
) -> core.Bundle:
    """Creates a new column with vectors that contain the two attributes"""
    b = b.copy()
    df = b.dfs[table_name]
    df[new_name] = list(zip(df[attribute1], df[attribute2]))
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - attribute1: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - attribute2: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
