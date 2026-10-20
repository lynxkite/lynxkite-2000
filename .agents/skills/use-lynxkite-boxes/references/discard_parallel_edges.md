**Discard parallel edges:**

```python
@op("Discard parallel edges", icon="filter-filled")
def discard_parallel_edges(graph: nx.Graph):
    return nx.DiGraph(graph)

```
