**Derive property:**

```python
@op("Derive property", icon="arrow-big-right-lines")
def derive_property(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.eval(formula)
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - formula: typing.Annotated[str, {'format': 'textarea'}]
