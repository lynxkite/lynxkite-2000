**Export to file:**
Exports a DataFrame to a file.
```python
@op("Export to file", icon="file-filled")
def export_to_file(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    filename: ops.PathStr,
    file_format: ExportFileFormat = ExportFileFormat.csv,
):
    """Exports a DataFrame to a file.

    Args:
        bundle: The bundle containing the DataFrame to export.
        table_name: The name of the DataFrame in the bundle to export.
        filename: The name of the file to export to.
        file_format: The format of the file to export to. Defaults to CSV.
    """

    df = bundle.dfs[table_name]
    if file_format == ExportFileFormat.csv:
        df.to_csv(filename, index=False)
    elif file_format == ExportFileFormat.json:
        df.to_json(filename, orient="records", lines=True)
    elif file_format == ExportFileFormat.parquet:
        df.to_parquet(filename, index=False)
    elif file_format == ExportFileFormat.excel:
        df.to_excel(filename, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - filename: typing.Annotated[str, {'format': 'path'}]
