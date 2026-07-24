**Import file:**
Read the contents of the a file into a `Bundle`.
```python
@op(
    "Import file",
    color="green",
    icon="file-filled",
    params=[
        ops.ParameterGroup(
            name="file_format_group",
            selector=ops.Parameter(name="file_format", type=FileFormat, default=FileFormat.csv),
            groups={
                "csv": [
                    ops.Parameter.basic("columns", type=str, default="<from file>"),
                    ops.Parameter.basic("separator", type=str, default="<auto>"),
                ],
                "parquet": [],
                "json": [],
                "excel": [ops.Parameter.basic("sheet_name", type=str, default="Sheet1")],
                "cif": [],
            },
            default=FileFormat.csv,
        ),
    ],
    slow=True,
)
def import_file(
    *, file_path: ops.PathStr, table_name: str, file_format: FileFormat = FileFormat.csv, **kwargs
) -> core.Bundle:
    """Read the contents of the a file into a `Bundle`.

    Args:
        file_path: Path to the file to import.
        table_name: Name to use for identifying the table in the bundle.
        file_format: Format of the file. Has to be one of the values in the `FileFormat` enum.

    Returns:
        Bundle: Bundle with a single table with the contents of the file.
    """
    if file_format == "csv":
        names = kwargs.get("columns", "<from file>")
        names = pd.api.extensions.no_default if names == "<from file>" else names.split(",")
        sep = kwargs.get("separator", "<auto>")
        sep = pd.api.extensions.no_default if sep == "<auto>" else sep.replace("\\t", "\t")
        df = pd.read_csv(file_path, names=names, sep=sep)  # ty: ignore[invalid-argument-type]
    elif file_format == "json":
        with open(file_path, "r") as f:
            df = pd.read_json(f)
    elif file_format == "parquet":
        df = pd.read_parquet(file_path)
    elif file_format == "excel":
        df = pd.read_excel(file_path, sheet_name=kwargs.get("sheet_name", "Sheet1"))
    elif file_format == "cif":
        with open(file_path, "r") as f:
            df = pd.DataFrame({"cif": [f.read()]})
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return core.Bundle(dfs={table_name: df})

```
Custom types:
  - file_path: typing.Annotated[str, {'format': 'path'}]
