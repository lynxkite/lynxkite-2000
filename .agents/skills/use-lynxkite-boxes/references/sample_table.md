**Sample table:**

```python
@op("Sample table", icon="filter-filled")
def sample_table(b: core.Bundle, *, table_name: core.TableName = "meta", fraction: float = 0.1):
    b = b.copy()
    b.dfs[table_name] = b.dfs[table_name].sample(frac=fraction)
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
