**Derive with SQL:**
Derives a new column with a SQL expression and stores it in the same table.
```python
@op("Derive with SQL", icon="database-plus")
def derive_with_sql(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    formula: ops.LongStr,
    name: str,
) -> core.Bundle:
    """
    Derives a new column with a SQL expression and stores it in the same table.
    :param b: the bundle.
    :param table_name: the name of the table to derive the column in.
    :param formula: the formula to derive the column with.
    :param name: the name of the derived column.
    """
    b = b.copy()
    query = f"select *, {formula} as {name} from {table_name}"
    b.dfs[table_name] = pl.SQLContext(b.dfs).execute(query).collect().to_pandas()
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - formula: typing.Annotated[str, {'format': 'textarea'}]
