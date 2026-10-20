**Degree:**

```python
@op("Degree", icon="topology-star-3")
def degree(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    nx.set_node_attributes(g, name="in_degree", values=dict(g.in_degree()))
    nx.set_node_attributes(g, name="out_degree", values=dict(g.out_degree()))
    nx.set_node_attributes(g, name="degree", values=dict(g.degree()))
    return g

```
