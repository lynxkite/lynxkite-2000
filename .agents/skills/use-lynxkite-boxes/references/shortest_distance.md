**Distance via shortest path:**
Computes the shortest distance from each node to the starting nodes using the specified edge distances.
```python
@op("Distance via shortest path", icon="link")
def shortest_distance(
    b: core.Bundle,
    *,
    relation: core.RelationName,
    edge_distances: str,
    attribute_name: str,
    starting_distance: str,
    max_iterations: str,
    undirected: bool,
) -> core.Bundle:
    """
    Computes the shortest distance from each node to the starting nodes using the specified edge distances.
    :param b: the bundle
    :param relation: the relation to use for the graph
    :param edge_distances: the distances for the edges
    :param attribute_name: the name of the attribute for storing the shortest distances
    :param starting_distance: the name of the attribute for the starting distances
    :param max_iterations: the maximum number of iterations allowed
    :param undirected: whether to treat the graph as undirected or not
    """
    b = b.copy()

    for r in b.relations:
        if r.name == relation:
            edge_df = b.dfs[r.df].copy()
            source_table, source_col, target_col, source_key = (
                r.source_table,
                r.source_column,
                r.target_column,
                r.source_key,
            )
            break
    else:
        raise ValueError(f"Relation '{relation}' not found.")

    edge_df[source_col] = edge_df[source_col].astype(str).str.strip()
    edge_df[target_col] = edge_df[target_col].astype(str).str.strip()

    if undirected:
        reverse = edge_df.rename(columns={source_col: target_col, target_col: source_col})
        edge_df = pd.concat([edge_df, reverse], ignore_index=True)

    nodes = b.dfs[source_table].copy()
    nodes[source_key] = nodes[source_key].astype(str).str.strip()
    nodes = nodes.set_index(source_key, drop=False)
    nodes[attribute_name] = pd.to_numeric(nodes[starting_distance], errors="coerce")

    for _ in range(int(max_iterations)):
        current = nodes[attribute_name].dropna()
        if current.empty:
            break

        merged = edge_df.merge(current, left_on=source_col, right_index=True, how="inner")
        if merged.empty:
            break

        merged["_candidate"] = merged[attribute_name] + merged[edge_distances]
        best = merged.groupby(target_col)["_candidate"].min()

        before = nodes[attribute_name].copy()
        nodes = nodes.join(best.rename("_candidate"), how="left")
        nodes[attribute_name] = nodes[[attribute_name, "_candidate"]].min(axis=1, skipna=True)
        nodes.drop(columns="_candidate", inplace=True)

        if nodes[attribute_name].equals(before):
            break

    node_lookup = b.dfs[source_table][source_key].astype(str).str.strip()
    b.dfs[source_table][attribute_name] = node_lookup.map(nodes[attribute_name])

    return b

```
Custom types:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
