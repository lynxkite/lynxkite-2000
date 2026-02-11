"""SQL and Cypher."""

import os
from lynxkite_core import ops

from .. import core
import grandcypher
import pandas as pd
import polars as pl


op = ops.op_registration(core.ENV)


@op("SQL", icon="database")
def sql(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "result"):
    """Run a SQL query on the DataFrames in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    if os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").strip().lower() == "true":
        with pl.Config() as cfg:
            cfg.set_verbose(True)
            res = pl.SQLContext(bundle.dfs).execute(query).collect(engine="gpu").to_pandas()
            # TODO: Currently `collect()` moves the data from cuDF to Polars. Then we convert it to Pandas,
            # which (hopefully) puts it back into cuDF. Hopefully we will be able to keep it in cuDF.
    else:
        res = pl.SQLContext(bundle.dfs).execute(query).collect().to_pandas()
    bundle.dfs[save_as] = res
    return bundle


@op("Cypher", icon="topology-star-3")
def cypher(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "result"):
    """Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    graph = bundle.to_nx()
    res = grandcypher.GrandCypher(graph).run(query)
    bundle.dfs[save_as] = pd.DataFrame(res)
    return bundle
