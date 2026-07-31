**Steiner forest:**
Creates a new dataframe that defines a Steiner Forest of the graph defined by the relation
```python
@op("Steiner forest", icon="binary-tree", slow=True)
def pcsf(
    b: core.Bundle,
    *,
    relation: core.RelationName,
    price_column: str,
    weight_column: str,
    root_cost_column: str,
    output_edge: str,
    output_node: str,
    output_root_nodes: str,
    output_profit: str,
):
    """
    Creates a new dataframe that defines a Steiner Forest of the graph defined by the relation
    :param b: the bundle
    :param relation: the relation
    :param price_column: the column with the node prices
    :param weight_column: the column with the edge weights
    :param root_cost_column: the column with the root costs
    :param output_edge: the output column, 1.0 if the edge is part of the forest, None otherwise
    :param output_node: the output column, 1.0 if the node is part of the forest, None otherwise
    :param output_root_nodes: the output column, 1.0 if the node is a root node, None otherwise
    :param output_profit: a table with a single record: the profit

    """
    b = b.copy()
    rel = next((r for r in b.relations if r.name == relation))
    if rel.source_table != rel.target_table:
        raise ValueError("Source and target tables must be the same.")

    node_df, edge_df = b.dfs[rel.source_table].copy(), b.dfs[rel.df].copy()
    nid, src, dst = rel.source_key, rel.source_column, rel.target_column

    node_df[nid] = node_df[nid].astype(str)
    edge_df[[src, dst]] = edge_df[[src, dst]].astype(str)

    node_df[price_column] = pd.to_numeric(node_df[price_column]).fillna(0.0).clip(lower=0.0)
    edge_df[weight_column] = pd.to_numeric(edge_df[weight_column]).fillna(0.0).clip(lower=0.0)

    raw_root_costs = pd.to_numeric(node_df[root_cost_column])
    eligible_root_mask = raw_root_costs.notna() & (raw_root_costs >= 0)
    eligible_root_nodes = set(node_df.loc[eligible_root_mask, nid])
    node_df["root_cost_sanitized"] = raw_root_costs.fillna(0.0).clip(lower=0.0)

    nodes = list(node_df[nid].unique())
    node_prices = dict(zip(node_df[nid], node_df[price_column]))
    root_costs = dict(zip(node_df[nid], node_df["root_cost_sanitized"]))

    undirected_edges = {}
    edge_costs = {}

    for idx, row in edge_df.iterrows():
        u, v, w = row[src], row[dst], row[weight_column]
        if u == v or u not in node_prices or v not in node_prices:
            continue

        key = tuple(sorted([u, v]))
        if key not in edge_costs or w < edge_costs[key]:
            undirected_edges[key] = idx
            edge_costs[key] = w

    edges = list(undirected_edges.keys())

    net_profit, selected_nodes, selected_roots, selected_edges = _gw_pcsf(
        nodes=nodes,
        und_list=edges,
        node_prices=node_prices,
        edge_costs=edge_costs,
        root_costs=root_costs,
        eligible_root_nodes=eligible_root_nodes,
    )

    selected_edge_indices = {undirected_edges[e] for e in selected_edges if e in undirected_edges}

    node_df[output_node] = [1.0 if x in selected_nodes else None for x in node_df[nid]]
    node_df[output_root_nodes] = [1.0 if x in selected_roots else None for x in node_df[nid]]
    edge_df[output_edge] = [1.0 if idx in selected_edge_indices else None for idx in edge_df.index]

    results_df = pd.DataFrame(
        {output_profit: [float(net_profit) if net_profit is not None else None]}
    )
    b.dfs[output_profit] = results_df

    if "root_cost_sanitized" in node_df.columns:
        node_df.drop(columns=["root_cost_sanitized"], inplace=True)

    b.dfs[rel.source_table] = node_df
    b.dfs[rel.df] = edge_df
    return b

```
Custom types:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
