**Drop tables:**
Keeps/removes the selected tables based on the value of drop_selected
```python
@op("Drop tables", color="orange", icon="table-minus")
def drop_tables(b: core.Bundle, *, keep_selected: bool, tables: core.MultiTableName) -> core.Bundle:
    """
    Keeps/removes the selected tables based on the value of drop_selected
    :param b: the bundle
    :param keep_selected: if False, removes the selected tables, otherwise the unselected ones
    :param tables: the tables to keep/remove
    """
    b = b.copy()

    b.dfs = {k: v for k, v in b.dfs.items() if (k in tables) == keep_selected}
    b.relations = [r for r in b.relations if r.source_table in b.dfs and r.target_table in b.dfs]
    return b

```
Custom types:
  - tables: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
