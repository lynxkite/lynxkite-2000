**Full prediction:**
Warning: This prediction can be a very expensive operation!
```python
@op("Full prediction", slow=True, color="yellow", icon="sparkles")
def full_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    k: int | None = None,
    inductive_setting: bool = False,
):
    """
    Warning: This prediction can be a very expensive operation!

    Args:
        k: Pass "" to keep all scores
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_all(
        model=model_wrapper, batch_size=None, k=k, mode="validation" if inductive_setting else None
    )
    pack = pred.process(
        factory=TriplesFactory(
            DUMMY_TRIPLET,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )
    bundle.dfs["pred"] = pack.df[
        ["head_label", "relation_label", "tail_label", "score"]
    ].sort_values(by="score", ascending=False)

    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
