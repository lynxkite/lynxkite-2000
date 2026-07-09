**Select Table:**

```python
@op("Select Table", icon="table-filled")
def select_table(b: core.Bundle, *, table_name: core.TableName) -> core.Bundle:
    df = b.dfs[table_name]
    bundle = core.Bundle()
    bundle.dfs[table_name] = df
    return bundle

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
