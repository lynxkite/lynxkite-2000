**Enter table data:**
Enter table data as CSV. The first row should contain column names.
```python
@op("Enter table data", color="green", icon="table-filled")
def enter_table_data(
    *,
    table_name: str,
    data: ops.LongStr,
):
    """Enter table data as CSV. The first row should contain column names."""
    b = core.Bundle()
    b.dfs[table_name] = pd.read_csv(io.StringIO(data))
    return b

```
Custom types:
  - data: typing.Annotated[str, {'format': 'textarea'}]
