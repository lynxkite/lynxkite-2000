**Remote SPARQL query:**
Run a SPARQL query on the given endpoint. For various knowledge graphs with SPARQL endpoints, see https://registry.okn.us/registry/
```python
@op("Remote SPARQL query", slow=True)
def remote_sparql(*, endpoint: str = "", query: ops.LongStr, name: str = "results"):
    """
    Run a SPARQL query on the given endpoint. For various knowledge graphs with SPARQL endpoints, see https://registry.okn.us/registry/
    :param endpoint: The SPARQL endpoint to send the query to.
    :param query: The SPARQL query to execute.
    :param name: The name of the resulting DataFrame.
    """
    r = requests.post(
        endpoint,
        data={"query": query},
        headers={
            "Accept": "application/sparql-results+json, text/turtle",
            "User-Agent": "LynxKite (https://lynxkite.com/; info@lynxkite.com)",
        },
    )
    r.raise_for_status()

    bundle = core.Bundle()
    if "json" in r.headers.get("Content-Type", ""):
        data = r.json()
        if "results" in data:
            rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
            bundle.dfs[name] = pd.DataFrame(rows, columns=data["head"]["vars"])
        else:
            bundle.dfs[name] = pd.DataFrame({"result": [data["boolean"]]})
    else:
        bundle.other[name] = Graph().parse(data=r.text, format="turtle")
    return bundle

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
