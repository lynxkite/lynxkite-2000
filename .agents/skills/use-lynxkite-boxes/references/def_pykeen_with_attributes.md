**Define PyKEEN model with node attributes:**
Defines a PyKEEN model capable of using numeric literals as node attributes.
```python
@op(
    "Define PyKEEN model with node attributes",
    color="green",
    icon="file-3d",
    params=[
        ops.ParameterGroup(
            name="combination_group",
            selector=ops.Parameter(
                name="combination_name",
                type=PyKEENCombinations,
                default=PyKEENCombinations.ConcatProjection,
            ),
            groups={
                "ComplexSeparated": [],
                # "Concat": [
                #     ops.Parameter.basic(name="dim", type=int, default=-1),
                # ],
                # "ConcatAggregation": [],
                "ConcatProjection": [
                    ops.Parameter.basic(name="bias", type=bool, default=False),
                    ops.Parameter.basic(name="dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="activation", type=str, default="ReLU"),
                ],
                "Gated": [
                    ops.Parameter.basic(name="input_dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="gate_activation", type=str, default="Sigmoid"),
                    ops.Parameter.basic(name="hidden_activation", type=str, default="Tanh"),
                ],
            },
            default=PyKEENCombinations.ConcatProjection,
        )
    ],
)
def def_pykeen_with_attributes(
    dataset: core.Bundle,
    *,
    interaction_name: PyKEENModel1D = PyKEENModel1D.TransE,
    combination_name: PyKEENCombinations = PyKEENCombinations.ConcatProjection,
    embedding_dim: int,
    loss_function: str,
    random_seed: int,
    save_as: str,
    **kwargs,
) -> core.Bundle:
    """Defines a PyKEEN model capable of using numeric literals as node attributes."""
    dataset = dataset.copy()

    edges_data = dataset.dfs["edges"][["head", "relation", "tail"]].astype(str)
    triples_no_literals = prepare_triples(
        edges_data,
    )
    temp_model = interaction_name.to_class(
        triples_factory=triples_no_literals,
        loss_func=loss_function,
        embedding_dim=embedding_dim,
        seed=random_seed,
    )

    num_literals = dataset.dfs["literals"]
    if "node_id" not in num_literals.columns:
        raise ValueError("Expected a 'node_id' column in literals DataFrame.")
    num_literals["node_id"] = num_literals["node_id"].astype(str)
    order = [
        label for label, _ in sorted(triples_no_literals.entity_to_id.items(), key=lambda kv: kv[1])
    ]
    num_literals = num_literals.set_index("node_id").reindex(order)

    if num_literals.isna().any().any():
        raise ValueError("Some entities are missing literals after reindexing.")

    features = num_literals.reset_index(drop=True)

    dataset.dfs["literals"] = num_literals
    literals_to_id = {label: i for i, label in enumerate(features.columns)}

    combination_cls = combination_name.to_class(embedding_dim, len(features.columns), **kwargs)

    assert isinstance(temp_model, models.ERModel), "Only models derived from ERModel are supported."
    try:
        interaction: Interaction = temp_model.interaction
    except AttributeError as e:
        raise Exception(
            "Interaction not supported for this model type. Please use a different interaction."
        ) from e

    model = LiteralModel(
        triples_factory=TriplesNumericLiteralsFactory(
            mapped_triples=triples_no_literals.mapped_triples,
            entity_to_id=triples_no_literals.entity_to_id,
            relation_to_id=triples_no_literals.relation_to_id,
            numeric_literals=torch.from_numpy(features.to_numpy())
            .contiguous()
            .detach()
            .cpu()
            .numpy(),
            literals_to_id=literals_to_id,
        ),
        entity_representations_kwargs=dict(
            shape=embedding_dim,
        ),
        relation_representations_kwargs=dict(
            shape=embedding_dim,
        ),
        interaction=interaction,  # ty: ignore[invalid-argument-type]
        combination=combination_cls,
        loss=loss_function,
        random_seed=random_seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        loss=loss_function,
        interaction=model.interaction,
        combination=combination_name,
        combination_kwargs=kwargs,
        literals_data=features,
        embedding_dim=embedding_dim,
        entity_to_id=triples_no_literals.entity_to_id,
        relation_to_id=triples_no_literals.relation_to_id,
        edges_data=edges_data,
        seed=random_seed,
    )

    dataset.other[save_as] = model_wrapper
    return dataset

```
