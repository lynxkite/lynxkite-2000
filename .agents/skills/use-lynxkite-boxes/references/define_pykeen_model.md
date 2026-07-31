**Define PyKEEN model:**
Defines a PyKEEN model based on the selected model type.
```python
@op("Define PyKEEN model", color="green", icon="file-3d")
def define_pykeen_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModelMoreD = PyKEENModelMoreD.MuRE,
    edge_data_table: core.TableName = "edges",
    embedding_dim: int = 50,
    loss_function: PyKEENSupportedLosses = PyKEENSupportedLosses.NSSALoss,
    seed: int = 42,
    save_as: str = "PyKEENmodel",
):
    """Defines a PyKEEN model based on the selected model type."""
    bundle = bundle.copy()
    edges_data = bundle.dfs[edge_data_table][["head", "relation", "tail"]]
    triples_factory = prepare_triples(
        edges_data,
        inv_triples=req_inverse_triples(model),
    )

    model_class = model.to_class(
        triples_factory=triples_factory,
        loss_func=loss_function,
        embedding_dim=embedding_dim,
        seed=seed,
    )
    model_wrapper = PyKEENModelWrapper(
        model_class,
        loss=loss_function,
        model_type=model,
        embedding_dim=embedding_dim,
        entity_to_id=triples_factory.entity_to_id,
        relation_to_id=triples_factory.relation_to_id,
        edges_data=edges_data,
        seed=seed,
    )
    bundle.other[save_as] = model_wrapper
    return bundle

```
Custom types:
  - edge_data_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
