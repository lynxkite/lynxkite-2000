**Train inductive model:**

```python
@op("Inductive setting", "Train inductive model", slow=True, color="purple", icon="barbell-filled")
def train_inductive_pykeen_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    transductive_table_name: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle_copy = bundle.copy()
    for key, value in bundle.dfs.items():
        bundle_copy.dfs[key] = value.copy(deep=True)

    model_wrapper: PyKEENModelWrapper = bundle_copy.other.get(model_name)
    bundle_copy.other[model_name] = model_wrapper.copy(deep=True)
    model_wrapper = bundle_copy.other[model_name]

    model = model_wrapper.model
    transductive_training = prepare_triples(
        bundle_copy.dfs[transductive_table_name][["head", "relation", "tail"]],
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
    )
    inductive_inference = prepare_triples(
        bundle_copy.dfs[inductive_inference_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
    )
    inductive_validation = prepare_triples(
        bundle_copy.dfs[inductive_validation_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
    )
    training_loop_cls = (
        SLCWATrainingLoop if training_approach == TrainingType.sLCWA else LCWATrainingLoop
    )
    loop_kwargs = (
        dict(
            negative_sampler_kwargs=dict(num_negs_per_pos=32),
        )
        if training_approach == TrainingType.sLCWA
        else dict()
    )
    training_loop = training_loop_cls(
        triples_factory=transductive_training,
        model=model,
        optimizer=optimizer_type,
        mode="training",
        **loop_kwargs,
    )

    valid_evaluator = evaluation.SampledRankBasedEvaluator(
        mode="validation",
        evaluation_factory=inductive_validation,
        additional_filter_triples=inductive_inference.mapped_triples,
    )

    early_stopper = stoppers.EarlyStopper(
        model=model,
        training_triples_factory=inductive_inference,
        evaluation_triples_factory=inductive_validation,
        frequency=5,
        patience=40,
        metric="ah@k",
        result_tracker=None,
        evaluation_batch_size=256,
        evaluator=valid_evaluator,
    )

    losses = training_loop.train(
        triples_factory=transductive_training,
        stopper=early_stopper,
        num_epochs=epochs,
    )

    model_wrapper.trained = True

    bundle_copy.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle_copy.dfs["early_stopper_metric"] = pd.DataFrame(
        {"early_stopper_metric": early_stopper.results}
    )
    bundle_copy.other[model_name] = model_wrapper

    return bundle_copy

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
  - transductive_table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
