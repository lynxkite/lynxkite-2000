"""Custom operations for the OKN knowledge graphs."""

import pandas as pd
import requests

from lynxkite_core import ops
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Knowledge graphs")


@op("SPARQL query", slow=True)
def sparql_query(*, endpoint: str, query: ops.LongStr):
    r = requests.post(
        endpoint,
        data={"query": query},
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "LynxKite (https://lynxkite.com/; info@lynxkite.com)",
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
    return pd.DataFrame(rows, columns=data["head"]["vars"])
