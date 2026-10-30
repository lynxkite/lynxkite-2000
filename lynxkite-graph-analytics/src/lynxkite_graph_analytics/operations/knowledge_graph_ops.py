"""Operations for working with knowledge graphs"""

import pandas as pd
import rdflib
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

    if "json" in r.headers.get("Content-Type", ""):
        data = r.json()
        if "results" in data:
            rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
            return pd.DataFrame(rows, columns=data["head"]["vars"])
        else:
            return pd.DataFrame({"result": [data["boolean"]]})
    else:
        return Graph().parse(data=r.text, format="turtle")


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
        return pd.DataFrame(list(result), columns=[str(v) for v in result.vars])
    elif result.type == "ASK":
        return pd.DataFrame({"result": [result.askAnswer]})
    else:
        return result.graph
