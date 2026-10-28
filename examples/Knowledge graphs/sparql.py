"""Custom operations for working with knowledge graphs with SPARQL endpoints."""

import pandas as pd
import requests

from lynxkite_core import ops
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Knowledge graphs")


@op("SPARQL query", slow=True)
def sparql_query(*, endpoint: str = "", query: ops.LongStr, name: str = "results"):
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
            "Accept": "application/sparql-results+json",
            "User-Agent": "LynxKite (https://lynxkite.com/; info@lynxkite.com)",
        },
    )
    r.raise_for_status()
    data = r.json()
    rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
    bundle = core.Bundle()
    bundle.dfs[name] = pd.DataFrame(rows, columns=data["head"]["vars"])
    return bundle
