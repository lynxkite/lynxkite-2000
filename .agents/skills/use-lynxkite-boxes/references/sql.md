**SQL:**
Run a SQL query on the DataFrames in the bundle. Save the results as a new DataFrame.
```python
@op("SQL", icon="database")
def sql(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "results"):
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

```
Custom types:
  - query: typing.Annotated[str, {'format': 'textarea'}]
