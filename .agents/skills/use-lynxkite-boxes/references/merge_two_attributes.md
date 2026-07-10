**Merge two attributes:**
An attribute may not be defined everywhere. This operation uses the secondary attribute to fill in the values where the primary attribute is undefined. If both are undefined then the result is undefined too.
```python
@op("Merge two attributes", icon="link")
def merge_two_attributes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    new_attribute: str,
    primary_attribute: core.ColumnNameByTableName,
    secondary_attribute: core.ColumnNameByTableName,
) -> core.Bundle:
    """
    An attribute may not be defined everywhere. This operation uses the secondary attribute to fill in the values where the primary attribute is undefined. If both are undefined then the result is undefined too.
    :param b: the bundle
    :param table_name: the name of the table
    :param new_attribute: the name of the new attribute
    :param primary_attribute: the primary attribute to use
    :param secondary_attribute: the secondary attribute to use
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    df[new_attribute] = df[primary_attribute].combine_first(df[secondary_attribute])
    b.dfs[table_name] = df
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - primary_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - secondary_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
