**Use table as attributes:**
Uses the columns from one table as attributes for the other.
```python
@op("Use table as attributes", icon="table-plus")
def table_as_attributes(
    bundle_graph: core.Bundle,
    bundle_att: core.Bundle,
    *,
    table_id: core.TableColumn,
    attribute_table_id: core.TableColumn,
    merge_mode: MergeMode,
) -> core.Bundle:
    """
    Uses the columns from one table as attributes for the other.
    :param bundle_graph: the bundle of the graph
    :param bundle_att: the bundle of the attributes
    :param table_id: the table that gets the attributes
    :param attribute_table_id: the table that provides the attributes
    :param merge_mode: determines what happens if an attribute already exists in the original table.
    Merge, prefer the table’s version: Where the table defines new values, those will be used. Elsewhere the existing values are kept.
    Merge, prefer the graph’s version: Where the vertex attribute is already defined, it is left unchanged. Elsewhere the value from the table is used.
    Merge, report error on conflict: An assertion is made to ensure that the values in the table are identical to the values in the graph on vertices where both are defined.
    Keep the graph’s version: The data in the table is ignored.
    Use the table’s version: The attribute is deleted from the graph and replaced with the attribute imported from the table.
    Disallow this: A name conflict is treated as an error.
    """
    bundle_graph = bundle_graph.copy()
    bundle_att = bundle_att.copy()

    graph_table = bundle_graph.dfs[table_id[0]].copy()
    attribute_table = bundle_att.dfs[attribute_table_id[0]].copy()
    merged = graph_table.merge(
        attribute_table,
        how="left",
        left_on=table_id[1],
        right_on=attribute_table_id[1],
        suffixes=("_graph", "_att"),
    )
    for column in attribute_table.columns:
        if column == attribute_table_id[1] or f"{column}_graph" not in merged.columns:
            continue

        if merge_mode == MergeMode.merge_table:
            merged[column] = merged[f"{column}_att"].combine_first(merged[f"{column}_graph"])
        elif merge_mode == MergeMode.merge_graph:
            merged[column] = merged[f"{column}_graph"].combine_first(merged[f"{column}_att"])
        elif merge_mode == MergeMode.report_conflict:
            conflict = merged[f"{column}_graph"] != merged[f"{column}_att"]
            if conflict.any():
                raise ValueError(f"Conflict in column {column}: {merged[conflict]}")
            merged[column] = merged[f"{column}_graph"].combine_first(merged[f"{column}_att"])
        elif merge_mode == MergeMode.use_table:
            merged[column] = merged[f"{column}_att"]
        elif merge_mode == MergeMode.keep_graph:
            merged[column] = merged[f"{column}_graph"]
        elif merge_mode == MergeMode.disallow:
            raise ValueError(f"Both tables have {column} column")
        merged.drop(columns=[f"{column}_graph", f"{column}_graph"], inplace=True)
    bundle_graph.dfs[table_id[0]] = merged
    return bundle_graph

```
Custom types:
  - table_id: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
  - attribute_table_id: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
