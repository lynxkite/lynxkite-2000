"""Operations for working with knowledge graphs"""

import pandas as pd
import requests
from rdflib import Graph
from lynxkite_core import ops
from .. import core

op = ops.op_registration(core.ENV, "Knowledge graphs")


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
