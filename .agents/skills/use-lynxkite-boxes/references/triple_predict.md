**Triples prediction:**

```python
@op("Triples prediction", color="yellow", icon="sparkles")
def triple_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    table_name: core.TableName = "edges_val",
    inductive_setting: bool = False,
):
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    actual_model = model.model
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    triples_to_predict_tf = prepare_triples(
        bundle.dfs[table_name][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        inv_triples=req_inverse_triples(actual_model) or inductive_setting,
    )

    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        original_repr = actual_model._get_entity_representations_from_inductive_mode(
            mode="validation"
        )
        actual_model = actual_model.to(torch.device("cpu"))
        actual_model.replace_entity_representations_(
            mode="validation",
            representation=actual_model.create_entity_representation_for_new_triples(
                triples_to_predict_tf
            ),
        )
        if torch.cuda.is_available():
            actual_model = actual_model.to(torch.device("cuda"))

    pred_df = (
        predict_triples(
            model=actual_model,
            triples_factory=triples_to_predict_tf,
            mode="validation" if inductive_setting else None,
        )
        .process(
            factory=TriplesFactory(
                DUMMY_TRIPLET,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
        )
        .df[["head_label", "relation_label", "tail_label", "score"]]
    )
    bundle.dfs["pred"] = pred_df
    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        # Restore the original entity representations after prediction
        actual_model.replace_entity_representations_(
            mode="validation", representation=original_repr
        )
    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
