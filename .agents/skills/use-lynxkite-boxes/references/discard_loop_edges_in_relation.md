**Discard loop edges in relation:**
Discards loop edges in the specified relation.
```python
@op("Discard loop edges in relation", icon="circle-off")
def discard_loop_edges_in_relation(b: core.Bundle, *, relation: core.RelationName):
    """
    Discards loop edges in the specified relation.
    :param b: the bundle
    :param relation: the relation
    """
    b = b.copy()
    for r in b.relations:
        if r.name == relation:
            df = b.dfs[r.df].copy()
            b.dfs[r.df] = df[df[r.source_column] != df[r.target_column]]
            break
    return b

```
Custom types:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
