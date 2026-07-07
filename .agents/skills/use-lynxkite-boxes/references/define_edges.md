**Define Edges:**
Define edges between node tables
```python
@op("Define Edges", view="graph_creation_view", outputs=["output"], icon="link")
def define_edges(b: core.Bundle, *, relations: str = ""):
    """Define edges between node tables"""
    b = b.copy()
    if relations.strip():
        new_relations = [core.RelationDefinition(**r) for r in json.loads(relations).values()]
        b.relations.extend(new_relations)
    return ops.Result(output=b, display=b.to_table_view(limit=100))

```
