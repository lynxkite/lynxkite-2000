**Import inductive dataset:**
Imports an inductive dataset from the PyKEEN library.
```python
@op("Inductive setting", "Import inductive dataset", color="green", icon="affiliate-filled")
def import_inductive_dataset(*, dataset: InductiveDataset = InductiveDataset.ILPC2022Small):
    """Imports an inductive dataset from the PyKEEN library."""
    ds = dataset.to_dataset()
    bundle = core.Bundle()
    bundle.dfs["transductive_training"] = pd.DataFrame(
        ds.transductive_training.triples, columns=["head", "relation", "tail"]
    )
    bundle.dfs["inductive_inference"] = pd.DataFrame(
        ds.inductive_inference.triples, columns=["head", "relation", "tail"]
    )
    bundle.dfs["inductive_testing"] = pd.DataFrame(
        ds.inductive_testing.triples, columns=["head", "relation", "tail"]
    )
    assert ds.inductive_validation is not None
    bundle.dfs["inductive_validation"] = pd.DataFrame(
        ds.inductive_validation.triples, columns=["head", "relation", "tail"]
    )
    return bundle

```
