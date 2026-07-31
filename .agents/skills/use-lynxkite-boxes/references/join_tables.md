**Join tables:**
Adds data from the second table to the first table.
```python
@op("Join tables", icon="table-plus")
def join_tables(
    b: core.Bundle,
    *,
    table1_column: core.TableColumn,
    table2_column: core.TableColumn,
    merge_mode: TableMergeMode,
) -> core.Bundle:
    """
    Adds data from the second table to the first table.
    :param b: the bundle
    :param table1_column: the first table and its column to join on
    :param table2_column: the second table and its column to join on
    :param merge_mode: determines what happens if a column is in both tables
    Merge, prefer the second table’s version: Where the second table defines values, those will be used. Elsewhere, the first table's values are used.
    Merge, prefer the first table’s version: Where the first table defines values, those will be used. Elsewhere, the second table's values are used.
    Merge, report error on conflict: An assertion is made to ensure that the values in the two tables are equal. If they are not, an error is raised.
    Use the first table’s version: The data in the second table is ignored.
    Use the second table’s version: The data in the first table is ignored.
    Only keep rows with matching values: Only rows that have matching values in both tables are kept.
    Keep both values with suffixes: Both values are kept, with suffixes added to the column names to distinguish them.
    Disallow this: A name conflict is treated as an error.
    """
    b = b.copy()
    primary_table = b.dfs[table1_column[0]].copy()
    secondary_table = b.dfs[table2_column[0]].copy()

    how = "inner" if merge_mode == TableMergeMode.only_matching else "left"

    merged = primary_table.merge(
        secondary_table,
        how=how,
        left_on=table1_column[1],
        right_on=table2_column[1],
        suffixes=("_1", "_2"),
    )

    for column in secondary_table.columns:
        if (
            column == table2_column[1]
            or f"{column}_1" not in merged.columns
            or merge_mode == TableMergeMode.both
        ):
            continue

        if merge_mode == TableMergeMode.merge_second:
            merged[column] = merged[f"{column}_1"].combine_first(merged[f"{column}_2"])
        elif merge_mode == TableMergeMode.merge_first:
            merged[column] = merged[f"{column}_2"].combine_first(merged[f"{column}_1"])
        elif merge_mode == TableMergeMode.report_conflict:
            conflict = merged[f"{column}_1"] != merged[f"{column}_2"]
            if conflict.any():
                raise ValueError(f"Conflict in column {column}: {merged[conflict]}")
            merged[column] = merged[f"{column}_1"].combine_first(merged[f"{column}_2"])
        elif merge_mode == TableMergeMode.use_second:
            merged[column] = merged[f"{column}_2"]
        elif merge_mode == TableMergeMode.use_first:
            merged[column] = merged[f"{column}_1"]
        elif merge_mode == TableMergeMode.only_matching:
            merged = merged[merged[f"{column}_1"] == merged[f"{column}_2"]]
            merged[column] = merged[f"{column}_1"]
        elif merge_mode == TableMergeMode.disallow:
            raise ValueError(f"Both tables have '{column}' column.")

        merged.drop(columns=[f"{column}_1", f"{column}_2"], inplace=True)

    b.dfs[table1_column[0]] = merged
    return b

```
Custom types:
  - table1_column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
  - table2_column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
