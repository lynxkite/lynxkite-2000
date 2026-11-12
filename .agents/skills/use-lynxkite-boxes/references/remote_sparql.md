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

    if "json" in r.headers.get("Content-Type", ""):
        b = core.Bundle()
        data = r.json()
        if "results" in data:
            rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
            df = pd.DataFrame(rows, columns=data["head"]["vars"])
        else:
            df = pd.DataFrame({"result": [data["boolean"]]})
        b.dfs[name] = df
        return b
    else:
        return Graph().parse(data=r.text, format="turtle")

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
