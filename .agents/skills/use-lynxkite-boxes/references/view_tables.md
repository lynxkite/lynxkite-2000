**View tables:**

```python
@op("View tables", view="table_view", color="blue", icon="table-filled")
def view_tables(bundle: core.Bundle, *, _tables_open: str = "", limit: int = 100):
    _tables_open = _tables_open  # The frontend uses this parameter to track which tables are open.
    return bundle.to_table_view(limit=limit)

```
