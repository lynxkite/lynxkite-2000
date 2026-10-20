**Import CSV:**
Imports a CSV file.
```python
@op("Import CSV", slow=True, color="green", icon="file-filled")
def import_csv(
    *,
    filename: ops.PathStr,
    columns: str = "<from file>",
    separator: str = "<auto>",
    table_name: str = "records",
):
    """Imports a CSV file."""
    names = pd.api.extensions.no_default if columns == "<from file>" else columns.split(",")
    sep = pd.api.extensions.no_default if separator == "<auto>" else separator
    df = pd.read_csv(filename, names=names, sep=sep)  # ty: ignore[invalid-argument-type]
    return core.Bundle(dfs={table_name: df})

```
Custom types:
  - filename: typing.Annotated[str, {'format': 'path'}]
