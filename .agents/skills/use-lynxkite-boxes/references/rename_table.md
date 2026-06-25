**Rename table:**
Assigns a new name to the table
```python
@op("Rename table", color="orange", icon="table-filled")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    return b

```
Custom types:
  - old_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
