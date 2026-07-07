**Join tables:**
Join/merge dataframes from two bundles.

Parameters:
- table_a: Table name from bundle A
- table_b: Table name from bundle B
- join_type: Type of join - "inner", "outer", "left", "right", "cross"
- on_column: Column name to join on (same name in both tables)
- left_on: Column name in left table (when column names differ)
- right_on: Column name in right table (when column names differ)
- suffixes: Suffixes for overlapping columns (comma-separated, e.g., "_a,_b")
```python
@op("Join tables", color="orange", icon="link")
def join_tables(
    bundle_a: core.Bundle,
    bundle_b: core.Bundle,
    *,
    table_a: core.TableName,
    table_b: core.TableName,
    join_type: JoinType = JoinType.inner,
    on_column: str = "",
    left_on: str = "",
    right_on: str = "",
    suffixes: str = "_a,_b",
):
    """
    Join/merge dataframes from two bundles.

    Parameters:
    - table_a: Table name from bundle A
    - table_b: Table name from bundle B
    - join_type: Type of join - "inner", "outer", "left", "right", "cross"
    - on_column: Column name to join on (same name in both tables)
    - left_on: Column name in left table (when column names differ)
    - right_on: Column name in right table (when column names differ)
    - suffixes: Suffixes for overlapping columns (comma-separated, e.g., "_a,_b")
    """

    df_a = bundle_a.dfs[table_a]
    df_b = bundle_b.dfs[table_b]

    # Parse suffixes
    suffix_parts = [s.strip() for s in suffixes.split(",")]
    if len(suffix_parts) != 2:
        suffix_list: tuple[str, str] = ("_a", "_b")
    else:
        suffix_list = (suffix_parts[0], suffix_parts[1])

    # Perform the join
    if on_column:
        merged_df = pd.merge(df_a, df_b, on=on_column, how=join_type.value, suffixes=suffix_list)
    elif left_on and right_on:
        merged_df = pd.merge(
            df_a,
            df_b,
            left_on=left_on,
            right_on=right_on,
            how=join_type.value,
            suffixes=suffix_list,
        )
    else:
        # Join on index if no columns specified
        merged_df = pd.merge(
            df_a, df_b, left_index=True, right_index=True, how=join_type.value, suffixes=suffix_list
        )

    return core.Bundle(dfs={"merged": merged_df})

```
Custom types:
  - table_a: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - table_b: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
