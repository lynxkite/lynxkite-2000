**Rename table:**
Assigns a new name to the table
```python
@op("Rename table", color="orange", icon="writing")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    relations = []
    for r in b.relations:
        r = r.copy()
        if r.source_table == old_name:
            r.source_table = new_name
        if r.target_table == old_name:
            r.target_table = new_name
        relations.append(r)
    b.relations = relations
    return b

```
Custom types:
  - old_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
