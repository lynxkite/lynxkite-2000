**Rename columns:**
Renames columns in the specified table according to the provided pairs of old and new names.
```python
@op("Rename columns", color="orange", icon="writing")
def rename_columns(
    b: core.Bundle, *, table_name: core.TableName, pairs: core.DropdownTextAdderByTableName
) -> core.Bundle:
    """
    Renames columns in the specified table according to the provided pairs of old and new names.
    :param b: the bundle.
    :param table_name: the table containing the columns to be renamed.
    :param pairs: the list of pairs (old_name, new_name).
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    for old_name, new_name in pairs:
        df.rename(columns={old_name: new_name}, inplace=True)
    b.dfs[table_name] = df
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - pairs: typing.Annotated[list[tuple[str, str]], {'format': 'dropdown-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}]
