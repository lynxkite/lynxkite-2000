**Cypher:**
Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame.
```python
@op("Cypher", icon="topology-star-3")
def cypher(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "results"):
    """Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    graph = bundle.to_nx()
    res = grandcypher.GrandCypher(graph).run(query)
    bundle.dfs[save_as] = pd.DataFrame(res)
    return bundle

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
