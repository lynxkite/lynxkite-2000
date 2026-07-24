**Graph from edge list:**

```python
@op("Graph from edge list", icon="topology-star-3")
def graph_from_edge_list(
    df: pd.DataFrame, *, source: core.RecordsColumn, target: core.RecordsColumn
) -> core.Bundle:
    b = core.Bundle()
    b.dfs["nodes"] = pd.DataFrame({"id": pd.concat([df[source], df[target]]).unique()})
    b.dfs["edges"] = df.rename(columns={source: "source", target: "target"})
    b.relations.append(
        core.RelationDefinition(
            name="graph",
            df="edges",
            source_column="source",
            source_table="nodes",
            source_key="id",
            target_column="target",
            target_table="nodes",
            target_key="id",
        )
    )
    return b

```
Custom types:
  - source: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}]
  - target: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}]
