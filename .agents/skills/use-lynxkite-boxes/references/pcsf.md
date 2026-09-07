**Steiner forest:**
The prize collecting Steiner tree is a problem that seeks a subtree of a graph that maximizes the total prize collected from the nodes minus the total weight of the edges.

The prize collecting Steiner Forest allows for multiple disjoint trees. This problem has multiple versions, in this case there are a set of nodes that can act as the root of the subtrees, and each such node has a cost for using it as the root of the tree it belongs to. Every subtree must have exactly 1 root.

A use case for this operation could be that we want to create a water supply network, where the water stations can act as the roots, and the houses have prizes, since they are the customers. The piping costs will be the weights of the edges.

This example can be seen in the "In Bruges" workspace in "examples/Peters lessons".

A small example of the PCSF problem:

We have a graph, with 5 nodes: A, B, C, D, E.

The edges with their weights:
A-B: 10
B-C: 20
D-E: 40

The nodes with their prizes:
A: 0
B: 30
C: 40
E: 25

The potential roots with the costs:
A: 15
D: 35

The optimal solution:
nodes: A, B, C
edges: A-B, B-C
roots: A
profit: (0 + 30 + 40) - (10 + 20) - (15) = 25

This box provides an approximate solution for the PCSF problem, as it is NP-hard.
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
    The prize collecting Steiner tree is a problem that seeks a subtree of a graph that maximizes the total prize collected from the nodes minus the total weight of the edges.

    The prize collecting Steiner Forest allows for multiple disjoint trees. This problem has multiple versions, in this case there are a set of nodes that can act as the root of the subtrees, and each such node has a cost for using it as the root of the tree it belongs to. Every subtree must have exactly 1 root.

    A use case for this operation could be that we want to create a water supply network, where the water stations can act as the roots, and the houses have prizes, since they are the customers. The piping costs will be the weights of the edges.

    This example can be seen in the "In Bruges" workspace in "examples/Peters lessons".

    A small example of the PCSF problem:

    We have a graph, with 5 nodes: A, B, C, D, E.

    The edges with their weights:
    A-B: 10
    B-C: 20
    D-E: 40

    The nodes with their prizes:
    A: 0
    B: 30
    C: 40
    E: 25

    The potential roots with the costs:
    A: 15
    D: 35

    The optimal solution:
    nodes: A, B, C
    edges: A-B, B-C
    roots: A
    profit: (0 + 30 + 40) - (10 + 20) - (15) = 25

    This box provides an approximate solution for the PCSF problem, as it is NP-hard.

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
    rel = next(r for r in b.relations if r.name == relation)
    if rel.source_table != rel.target_table:
        raise ValueError("Source and target tables must be the same.")

    node_df, edge_df = b.dfs[rel.source_table].copy(), b.dfs[rel.df].copy()
    nid, src, dst = rel.source_key, rel.source_column, rel.target_column

    node_df[nid] = node_df[nid].astype(str)
    edge_df[[src, dst]] = edge_df[[src, dst]].astype(str)

    node_df[price_column] = pd.to_numeric(node_df[price_column]).fillna(0.0).clip(lower=0.0)
    edge_df[weight_column] = pd.to_numeric(edge_df[weight_column]).fillna(0.0).clip(lower=0.0)

    raw_root_costs = pd.to_numeric(node_df[root_cost_column])
    node_df["_root_cost"] = raw_root_costs.clip(lower=0.0)
    eligible_root_nodes = set(node_df.loc[raw_root_costs.notna() & (raw_root_costs >= 0), nid])

    node_prices = dict(zip(node_df[nid], node_df[price_column]))
    root_costs = dict(zip(node_df[nid], node_df["_root_cost"].fillna(0.0)))

    edge_df["_key"] = edge_df.apply(lambda r: tuple(sorted([r[src], r[dst]])), axis=1)
    edge_df = edge_df[edge_df[src] != edge_df[dst]]
    edge_df = edge_df[edge_df[src].isin(node_prices) & edge_df[dst].isin(node_prices)]
    cheapest = edge_df.groupby("_key")[weight_column].idxmin()
    undirected_edges = (
        edge_df.loc[cheapest, ["_key", weight_column]].set_index("_key")[weight_column].to_dict()
    )
    original_idx = edge_df.loc[cheapest, "_key"].reset_index().set_index("_key")["index"].to_dict()

    nodes = list(node_df[nid].unique())
    node_index = {node: i for i, node in enumerate(nodes)}
    virtual_root = len(nodes)

    solver_edges = [(node_index[u], node_index[v]) for u, v in undirected_edges]
    solver_costs = [float(w) for w in undirected_edges.values()]
    eligible_roots = [r for r in eligible_root_nodes if r in node_index]
    for root in eligible_roots:
        solver_edges.append((virtual_root, node_index[root]))
        solver_costs.append(float(root_costs.get(root, 0.0)))

    selected_nodes, selected_roots, selected_edges = set(), set(), set()
    net_profit = 0.0
    if solver_edges and eligible_roots:
        prizes = np.asarray(
            [float(node_prices.get(n, 0.0)) for n in nodes] + [0.0], dtype=np.float64
        )
        result_nodes, result_edges = pcst_fast.pcst_fast(
            np.asarray(solver_edges, dtype=np.int64),
            prizes,
            np.asarray(solver_costs, dtype=np.float64),
            virtual_root,
            1,
            "strong",
            0,
        )
        if virtual_root in result_nodes:
            selected_nodes = {str(nodes[i]) for i in result_nodes if i != virtual_root}
            for edge_id in result_edges:
                i, j = solver_edges[int(edge_id)]
                if virtual_root in (i, j):
                    selected_roots.add(str(nodes[j if i == virtual_root else i]))
                else:
                    u, v = str(nodes[i]), str(nodes[j])
                    selected_edges.add((u, v) if u < v else (v, u))
            net_profit = (
                sum(float(node_prices.get(n, 0.0)) for n in selected_nodes)
                - sum(float(undirected_edges[e]) for e in selected_edges)
                - sum(float(root_costs.get(r, 0.0)) for r in selected_roots)
            )
            if net_profit < 0.0:
                selected_nodes, selected_roots, selected_edges, net_profit = (
                    set(),
                    set(),
                    set(),
                    0.0,
                )

    selected_edge_indices = {original_idx[e] for e in selected_edges if e in original_idx}

    node_df[output_node] = node_df[nid].isin(selected_nodes).map({True: 1.0, False: None})
    node_df[output_root_nodes] = node_df[nid].isin(selected_roots).map({True: 1.0, False: None})
    edge_df[output_edge] = pd.Series(
        edge_df.index.isin(selected_edge_indices), index=edge_df.index
    ).map({True: 1.0, False: None})

    b.dfs[output_profit] = pd.DataFrame({output_profit: [net_profit]})
    node_df.drop(columns=["_root_cost"], inplace=True)
    edge_df.drop(columns=["_key"], inplace=True)

    b.dfs[rel.source_table] = node_df
    b.dfs[rel.df] = edge_df
    return b

```
Custom types:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
