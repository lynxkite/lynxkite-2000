**Local SPARQL query:**
Run a SPARQL query on a local rdf knowledge graph.
```python
@op("Local SPARQL query")
def local_sparql(bundle: core.Bundle, *, graph: str, query: ops.LongStr, name: str = "results"):
    """
    Run a SPARQL query on a local rdf knowledge graph.
    :param bundle: The bundle containing the knowledge graph.
    :param graph: The name of the knowledge graph within to query.
    :param query: The SPARQL query to execute.
    :param name: The name of the resulting DataFrame.
    """
    g = bundle.other[graph]
    result = g.query(query)

    if result.type == "SELECT":
        bundle.dfs[name] = pd.DataFrame(list(result), columns=[str(v) for v in result.vars])
    elif result.type == "ASK":
        bundle.dfs[name] = pd.DataFrame({"result": [result.askAnswer]})
    else:
        bundle.other[name] = result.graph
    return bundle

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
