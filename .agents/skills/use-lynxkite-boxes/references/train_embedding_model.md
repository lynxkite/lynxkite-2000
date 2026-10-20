**Train embedding model:**

```python
@op("Train embedding model", slow=True, color="purple", icon="barbell-filled")
def train_embedding_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModelName = "PyKEENmodel",
    training_table: core.TableName = "edges_train",
    testing_table: core.TableName = "edges_test",
    validation_table: core.TableName = "edges_val",
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    learning_rate: float = 0.0001,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
    number_of_negative_samples_per_positive: int = 512,
):
    bundle_copy = bundle.copy()
    for key, value in bundle.dfs.items():
        bundle_copy.dfs[key] = value.copy(deep=True)

    model_wrapper: PyKEENModelWrapper = bundle_copy.other.get(model)
    bundle_copy.other[model] = model_wrapper.copy(deep=True)
    model_wrapper = bundle_copy.other[model]
    actual_model = model_wrapper.model
    sampler = None
    if isinstance(actual_model, models.RGCN) and training_approach == TrainingType.sLCWA:
        # Currently RGCN is the only model that requires a sampler and only when using sLCWA
        sampler = "schlichtkrull"

    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id

    training_set = prepare_triples(
        bundle_copy.dfs[training_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    testing_set = prepare_triples(
        bundle_copy.dfs[testing_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    validation_set = prepare_triples(
        bundle_copy.dfs[validation_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    training_set, testing_set, validation_set = leakage.unleak(
        training_set, testing_set, validation_set
    )
    result: PipelineResult = pipeline(
        training=training_set,
        testing=testing_set,
        validation=validation_set,
        model=actual_model,
        loss=model_wrapper.loss,
        optimizer=optimizer_type,
        optimizer_kwargs=dict(
            lr=learning_rate,
        ),
        lr_scheduler="PolynomialLR",
        lr_scheduler_kwargs=dict(
            total_iters=epochs,
            power=0.95,
        ),
        training_loop=training_approach,
        negative_sampler="bernoulli" if training_approach == TrainingType.sLCWA else None,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=number_of_negative_samples_per_positive,
        ),
        epochs=epochs,
        training_kwargs=dict(
            sampler=sampler,
            continue_training=model_wrapper.trained,
        ),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=40,
            relative_delta=0.0005,
            metric="ah@k",
        ),
        random_seed=model_wrapper.seed,
    )

    model_wrapper.model = result.model
    model_wrapper.trained = True

    bundle_copy.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    if isinstance(result.stopper, stoppers.EarlyStopper):
        bundle_copy.dfs["early_stopper_metric"] = pd.DataFrame(
            {"early_stopper_metric": result.stopper.results}
        )
    bundle_copy.other[model] = model_wrapper

    return bundle_copy

```
Custom types:
  - model: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
  - training_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
