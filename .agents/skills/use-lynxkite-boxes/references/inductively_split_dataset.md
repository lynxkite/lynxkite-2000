**Split inductive dataset:**
Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
Inference testing and validation sets that can be used to evaluate model performance.
```python
@op("Inductive setting", "Split inductive dataset", color="orange", icon="circle-half-2")
def inductively_split_dataset(
    bundle: core.Bundle,
    *,
    dataset_table: core.TableName,
    entity_ratio: float = 0.5,
    training_ratio: float = 0.8,
    testing_ratio: float = 0.1,
    validation_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
    Inference testing and validation sets that can be used to evaluate model performance.

    Args:
        entity_ratio: How many percent of the entities in the dataset should be in the transductive training graph. If `0` semi-inductive split is applied, else fully-inductive split is applied
        training_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference training split
        testing_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference testing split
        validation_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference validation split
    """
    bundle = bundle.copy()

    bundle.dfs[dataset_table] = bundle.dfs[dataset_table].astype(str)
    tf_all = TriplesFactory.from_labeled_triples(
        bundle.dfs[dataset_table][["head", "relation", "tail"]].to_numpy(),
    )
    ratios = (training_ratio, testing_ratio, validation_ratio)
    if entity_ratio == 0:
        tf_training, tf_validation, tf_testing = tf_all.split_semi_inductive(
            ratios=ratios, random_state=seed
        )
    else:
        tf_training, tf_inference, tf_validation, tf_testing = tf_all.split_fully_inductive(
            entity_split_train_ratio=entity_ratio,
            evaluation_triples_ratios=ratios,
            random_state=seed,
        )
    transductive = pd.DataFrame(tf_training.triples, columns=["head", "relation", "tail"])
    inductive_testing = pd.DataFrame(tf_testing.triples, columns=["head", "relation", "tail"])
    inductive_val = pd.DataFrame(tf_validation.triples, columns=["head", "relation", "tail"])

    inductive_inference = (
        pd.concat([inductive_val, inductive_testing], ignore_index=True)
        if entity_ratio == 0
        else pd.DataFrame(tf_inference.triples, columns=["head", "relation", "tail"])
    )

    bundle.dfs["transductive_training"] = transductive
    bundle.dfs["inductive_inference"] = inductive_inference
    bundle.dfs["inductive_testing"] = inductive_testing
    bundle.dfs["inductive_validation"] = inductive_val

    return bundle

```
Custom types:
  - dataset_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
