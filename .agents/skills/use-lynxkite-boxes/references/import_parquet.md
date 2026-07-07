**Import Parquet:**
Imports a Parquet file.
```python
@op("Import Parquet", color="green", icon="file-filled")
def import_parquet(*, filename: ops.PathStr):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)

```
Custom types:
  - filename: typing.Annotated[str, {'format': 'path'}]
