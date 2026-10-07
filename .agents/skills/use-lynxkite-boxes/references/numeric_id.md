**Numeric ids:**
replaces the ids used in the relation with numeric ids
```python
@op("Numeric ids", color="green", icon="table-filled")
def numeric_id(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
):
    """replaces the ids used in the relation with numeric ids
    :param b: The bundle.
    :param relation_name: The name of the relation.
    """
    b = b.copy()
    b.dfs = b.dfs.copy()

    rel = next((r for r in b.relations if r.name == relation_name))

    source_df = b.dfs[rel.source_table].copy()
    target_df = b.dfs[rel.target_table].copy()
    edge_df = b.dfs[rel.name].copy()

    global_source_map = {old_id: idx for idx, old_id in enumerate(source_df[rel.source_key])}
    source_df[rel.source_key] = range(len(source_df))

    if rel.source_table == rel.target_table:
        global_target_map = global_source_map
        target_df[rel.target_key] = source_df[rel.source_key]
    else:
        global_target_map = {old_id: idx for idx, old_id in enumerate(target_df[rel.target_key])}
        target_df[rel.target_key] = range(len(target_df))

    edge_df[rel.source_column] = edge_df[rel.source_column].map(global_source_map)
    edge_df[rel.target_column] = edge_df[rel.target_column].map(global_target_map)

    b.dfs[rel.source_table] = source_df
    b.dfs[rel.target_table] = target_df
    b.dfs[rel.name] = edge_df

    for df_name, df in b.dfs.items():
        if df_name in (rel.source_table, rel.target_table, rel.name):
            continue

        if rel.source_key in df.columns:
            updated_df = df.copy()
            updated_df[rel.source_key] = updated_df[rel.source_key].map(global_source_map)
            b.dfs[df_name] = updated_df
    return b

```
Custom types:
  - relation_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
