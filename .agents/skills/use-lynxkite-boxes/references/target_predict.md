**Target prediction:**
Leave the target prediction field empty
```python
@op("Target prediction", color="yellow", icon="sparkles")
def target_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    head: str,
    relation: str,
    tail: str,
    inductive_setting: bool = False,
):
    """
    Leave the target prediction field empty
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_target(
        model=model_wrapper,
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory(
            DUMMY_TRIPLET,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
        mode="validation" if inductive_setting else None,
    )

    col = "head_label" if head == "" else "tail_label" if tail == "" else "relation_label"
    df = pred.df[[col, "score"]]

    bundle.dfs["pred"] = df
    bundle.dfs["pred"].sort_values(by="score", ascending=False, inplace=True)
    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
