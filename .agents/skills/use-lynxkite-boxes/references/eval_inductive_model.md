**Evaluate inductive model:**

```python
@op("Inductive setting", "Evaluate inductive model", color="orange", icon="microscope-filled")
def eval_inductive_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    inductive_testing_table: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    metrics_str: str = "ALL",
    batch_size: int = 32,
):
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    inductive_testing = prepare_triples(
        bundle.dfs[inductive_testing_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )
    inductive_inference = prepare_triples(
        bundle.dfs[inductive_inference_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )
    inductive_validation = prepare_triples(
        bundle.dfs[inductive_validation_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )

    test_evaluator = evaluation.SampledRankBasedEvaluator(
        mode="testing",
        evaluation_factory=inductive_testing,
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
    )

    result = test_evaluator.evaluate(
        model=model_wrapper,
        mapped_triples=inductive_testing.mapped_triples,
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
        batch_size=batch_size,
    )
    if metrics_str == "ALL":
        bundle.dfs["metrics"] = result.to_df()
        return bundle

    metrics = metrics_str.split(",")
    metrics_df = pd.DataFrame(columns=["metric", "score"])

    for metric in metrics:
        metric = metric.strip()
        try:
            score = result.get_metric(metric)
        except Exception as e:
            raise Exception(f"Possibly unknown metric: {metric}") from e
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([[metric, score]], columns=metrics_df.columns)]
        )

    bundle.dfs["metrics"] = metrics_df

    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
  - inductive_testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
