**Aggregate on neighbors:**

```python
@op("Aggregate on neighbors", icon="topology-star-3")
def aggregate_on_neighbors(
    g: nx.Graph, *, property: core.NodePropertyName, aggregation: AggregationMethod
) -> nx.Graph:
    g = g.copy()
    for node in g.nodes:
        neighbor_values = [g.nodes[neighbor].get(property, 0) for neighbor in g.neighbors(node)]
        if not neighbor_values:
            continue
        agg_value = aggregation.apply(neighbor_values)
        g.nodes[node][f"{property}_neighborhood_{aggregation}"] = agg_value
    return g

```
Custom types:
  - property: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}]
