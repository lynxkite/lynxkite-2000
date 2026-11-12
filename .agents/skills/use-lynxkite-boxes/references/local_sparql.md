**Local SPARQL query:**
Run a SPARQL query on a local rdf knowledge graph.
```python
@op("Local SPARQL query")
def local_sparql(graph: rdflib.Graph, *, query: ops.LongStr, name: str = "results"):
    """
    Run a SPARQL query on a local rdf knowledge graph.
    :param graph: The RDF graph to query.
    :param query: The SPARQL query to execute.
    :param name: The name of the resulting DataFrame.
    """
    result = graph.query(query)
    if result.type == "SELECT":
        df = pd.DataFrame(list(result), columns=[str(v) for v in result.vars])
    elif result.type == "ASK":
        df = pd.DataFrame({"result": [result.askAnswer]})
    else:
        return result.graph
    b = core.Bundle()
    b.dfs[name] = df
    return b

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
