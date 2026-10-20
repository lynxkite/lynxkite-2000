**Import GraphML:**
Imports a GraphML file.
```python
@op("Import GraphML", slow=True, color="green", icon="topology-star-3")
def import_graphml(*, filename: ops.PathStr):
    """Imports a GraphML file."""
    files = fsspec.open_files(filename, compression="infer")
    for f in files:
        if ".graphml" in f.path:
            with f as f:
                return nx.read_graphml(f)
    raise ValueError(f"No .graphml file found at {filename}")

```
Custom types:
  - filename: typing.Annotated[str, {'format': 'path'}]
