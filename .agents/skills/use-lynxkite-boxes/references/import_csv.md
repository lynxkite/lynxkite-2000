**Import CSV:**
Imports a CSV file.
```python
@op("Import CSV", slow=True, color="green", icon="file-filled")
def import_csv(*, filename: ops.PathStr, columns: str = "<from file>", separator: str = "<auto>"):
    """Imports a CSV file."""
    names = pd.api.extensions.no_default if columns == "<from file>" else columns.split(",")
    sep = pd.api.extensions.no_default if separator == "<auto>" else separator
    return pd.read_csv(filename, names=names, sep=sep)  # ty: ignore[invalid-argument-type]

```
Custom types:
  - filename: typing.Annotated[str, {'format': 'path'}]
