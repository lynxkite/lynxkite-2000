**Fill attributes with default values:**
An attribute may not be defined everywhere. This operation sets the provided values for the rows of the specified attributes where they are not defined.
```python
@op("Fill attributes with default values", icon="table-column")
def fill_with_default(
    b: core.Bundle, *, table_name: core.TableName, adder: core.DropdownTextAdderByTableName
) -> core.Bundle:
    """
    An attribute may not be defined everywhere. This operation sets the provided values for the rows of the specified attributes where they are not defined.
    :param b: the bundle
    :param table_name: the table to operate on
    :param adder: the attributes and the values to set
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    for column, default_value in adder:
        df[column] = df[column].fillna(default_value)
    b.dfs[table_name] = df
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - adder: typing.Annotated[list[tuple[str, str]], {'format': 'dropdown-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}]
