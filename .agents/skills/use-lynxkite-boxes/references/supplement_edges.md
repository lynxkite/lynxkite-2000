**Supplement edges with node attributes:**
Adds the attributes of the source and target nodes to the edges in the specified relation.
```python
@op("Supplement edges with node attributes", icon="link")
def supplement_edges(b: core.Bundle, *, table_name: core.TableName) -> core.Bundle:
    """
    Adds the attributes of the source and target nodes to the edges in the specified relation.
    :param b: the bundle
    :param table_name: the name of the edge table
    """
    b = b.copy()
    for r in b.relations:
        if r.df == table_name:
            df = b.dfs[table_name].copy()
            source_df = b.dfs[r.source_table].copy()
            target_df = b.dfs[r.target_table].copy()
            for src_column in source_df.columns:
                if src_column != r.source_key:
                    df[f"{src_column}_src"] = df[r.source_column].map(
                        source_df.set_index(r.source_key)[src_column]
                    )
            for tgt_column in target_df.columns:
                if tgt_column != r.target_key:
                    df[f"{tgt_column}_dst"] = df[r.target_column].map(
                        target_df.set_index(r.target_key)[tgt_column]
                    )
            b.dfs[r.df] = df
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
