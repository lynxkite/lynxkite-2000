**Define inductive PyKEEN model:**
Defines an InductiveNodePiece model (with an optional GNN message passing layer) for inductive link prediction tasks.
```python
@op("Inductive setting", "Define inductive PyKEEN model", color="green", icon="file-3d")
def get_inductive_model(
    bundle: core.Bundle,
    *,
    triples_table: core.TableName,
    inference_table: core.TableName,
    interaction: PyKEENModel1D = PyKEENModel1D.DistMult,
    embedding_dim: int = 200,
    loss_function: str,
    num_tokens: int = 2,
    aggregation: PyTorchAggregationFunctions = PyTorchAggregationFunctions.MLP,
    use_GNN: bool = False,
    seed: int = 42,
    save_as: str = "InductiveModel",
):
    """
    Defines an InductiveNodePiece model (with an optional GNN message passing layer) for inductive link prediction tasks.

    Args:
        triples_table: The transductive edges of the graph.
        inference_table: The inductive edges of the graph.
        interaction: Type of interaction the model will use for link prediction scoring.
        num_tokens: Number of hash tokens for each node representation, usually 66th percentiles of the number of unique incident relations per node.
        aggregation: Aggregation of multiple token representations to a single entity representation. Pick a top-level torch function, or use 'mlp' for a two-layer built-in mlp aggregator.
    """
    bundle = bundle.copy()
    transductive_training = prepare_triples(
        bundle.dfs[triples_table][["head", "relation", "tail"]],
        inv_triples=True,
    )
    inductive_inference = prepare_triples(
        bundle.dfs[inference_table][["head", "relation", "tail"]],
        relation_to_id=transductive_training.relation_to_id,
        inv_triples=True,
    )
    model_cls = models.InductiveNodePieceGNN if use_GNN else models.InductiveNodePiece
    base_model_cls = interaction.to_class(transductive_training, loss_function, embedding_dim, 42)
    assert isinstance(base_model_cls, models.ERModel), "Base model class is not an ERModel"
    interaction_cls = base_model_cls.interaction

    model = model_cls(
        triples_factory=transductive_training,
        inference_factory=inductive_inference,
        loss=loss_function,
        interaction=interaction_cls,
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,
        aggregation=aggregation,
        random_seed=seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        loss=loss_function,
        embedding_dim=embedding_dim,
        entity_to_id=inductive_inference.entity_to_id,
        relation_to_id=transductive_training.relation_to_id,
        edges_data=bundle.dfs[triples_table][["head", "relation", "tail"]],
        seed=seed,
        interaction=model.interaction,
        inductive_inference=bundle.dfs[inference_table][["head", "relation", "tail"]],
        inductive_kwargs=dict(
            num_tokens=num_tokens,
            aggregation=aggregation,
            use_GNN="True" if use_GNN else "False",
        ),
    )

    bundle.other[save_as] = model_wrapper
    return bundle

```
Custom types:
  - triples_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
