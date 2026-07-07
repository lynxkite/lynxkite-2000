**Add rank attribute:**
Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row
```python
@op("Add rank attribute", icon="link")
def add_rank(
    b: core.Bundle,
    *,
    table_column: core.TableColumn,
    rank_name: str,
    order: OrderType,
):
    """Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row

    Parameters
    ----------
    table_column : core.TableColumn
        The table and column to rank
    rank_name : str
        The name of the new rank column
    order : OrderType
        The order in which to rank the rows either 'ascending' or 'descending'

    Returns
    -------
    output : core.Bundle
        The updated bundle with the new rank column
    """
    table, column = table_column
    b = b.copy()
    df = b.dfs[table]

    df = df.sort_values(by=column, ascending=(order == OrderType.asc))
    df[rank_name] = range(len(df))

    b.dfs[table] = df
    return b

```
Custom types:
  - table_column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
