**Evaluate model:**
Evaluates the given model on the test set using the specified evaluator type.
Args:
    evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
    metrics_str: Comma separated list, "ALL" if all metrics are needed.
```python
@op("Evaluate model", slow=True, color="orange", icon="microscope-filled")
def evaluate(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    evaluator_type: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator,
    eval_table: core.TableName = "edges_test",
    additional_true_triples_table: core.TableName = "edges_train",
    metrics_str: str = "ALL",
    batch_size: int = 32,
):
    """
    Evaluates the given model on the test set using the specified evaluator type.
    Args:
        evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
        metrics_str: Comma separated list, "ALL" if all metrics are needed.
    """

    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    evaluator = evaluator_type.to_class()
    if isinstance(evaluator, evaluation.ClassificationEvaluator):
        from pykeen.metrics.classification import classification_metric_resolver

        evaluator.metrics = tuple(
            classification_metric_resolver.make(metric_cls) for metric_cls in metrics_str.split(",")
        )
    testing_triples = prepare_triples(
        bundle.dfs[eval_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    additional_filters = prepare_triples(
        bundle.dfs[additional_true_triples_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    evaluated = evaluator.evaluate(
        model=model_wrapper.model,
        mapped_triples=testing_triples.mapped_triples,
        additional_filter_triples=additional_filters.mapped_triples,
        batch_size=batch_size,
    )
    if metrics_str == "ALL":
        bundle.dfs["metrics"] = evaluated.to_df()
        return bundle

    metrics = metrics_str.split(",")
    metrics_df = pd.DataFrame(columns=["metric", "score"])

    for metric in metrics:
        metric = metric.strip()
        try:
            score = evaluated.get_metric(metric)
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
  - eval_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - additional_true_triples_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
