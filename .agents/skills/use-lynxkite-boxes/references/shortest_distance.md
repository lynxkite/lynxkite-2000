**Distance via shortest path:**
Computes the shortest distance from each node to the starting nodes using the specified edge distances.
```python
@op("Distance via shortest path", icon="route-square")
def shortest_distance(
    b: core.Bundle,
    *,
    relation: core.RelationName,
    edge_distances: str,
    attribute_name: str,
    starting_distance: str,
    undirected: bool,
) -> core.Bundle:
    """
    Computes the shortest distance from each node to the starting nodes using the specified edge distances.
    :param b: the bundle
    :param relation: the relation to use for the graph
    :param edge_distances: the distances for the edges
    :param attribute_name: the name of the attribute for storing the shortest distances
    :param starting_distance: the name of the attribute for the starting distances
    :param undirected: whether to treat the graph as undirected or not
    """
    b = b.copy()
    r = next(r for r in b.relations if r.name == relation)
    if r.source_table != r.target_table:
        raise ValueError("Source and target tables must be the same.")

    edges = b.dfs[r.df].copy()
    nodes = b.dfs[r.source_table]

    weight_col = "_weight"
    edges[weight_col] = pd.to_numeric(edges[edge_distances], errors="coerce").fillna(1.0)

    G = nx.from_pandas_edgelist(
        edges,
        source=r.source_column,
        target=r.target_column,
        edge_attr=[weight_col],
        create_using=nx.Graph if undirected else nx.DiGraph,
    )
    G.add_nodes_from(nodes[r.source_key])

    virtual_source = "_virtual_source_"
    valid_dists = pd.to_numeric(nodes[starting_distance], errors="coerce")

    virtual_edges = [
        (virtual_source, node_id, dist)
        for node_id, dist in zip(nodes[r.source_key], valid_dists)
        if pd.notna(dist)
    ]
    G.add_weighted_edges_from(virtual_edges, weight=weight_col)

    distances = nx.single_source_bellman_ford_path_length(
        G, source=virtual_source, weight=weight_col
    )
    distances.pop(virtual_source, None)
    b.dfs[r.source_table][attribute_name] = nodes[r.source_key].map(distances)

    return b

```
Custom types:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
