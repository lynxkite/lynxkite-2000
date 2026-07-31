**Import PyKEEN dataset:**
Imports a dataset from the PyKEEN library.
```python
@op("Import PyKEEN dataset", slow=True, color="green", icon="file-filled")
def import_pykeen_dataset_path(
    self, *, dataset: PyKEENDataset = PyKEENDataset.Nations
) -> core.Bundle:
    """Imports a dataset from the PyKEEN library."""
    ds = dataset.to_dataset()
    bundle = core.Bundle()

    bundle.dfs["edges_train"] = factory_to_df(factory=ds.training)
    bundle.dfs["edges_test"] = factory_to_df(factory=ds.testing)
    if ds.validation:
        bundle.dfs["edges_val"] = factory_to_df(factory=ds.validation)

    bundle.dfs["nodes"] = pd.DataFrame(
        {
            "id": list(ds.entity_to_id.values()),
            "label": list(ds.entity_to_id.keys()),
        }
    )
    bundle.dfs["relations"] = pd.DataFrame(
        {
            "id": list(ds.relation_to_id.values()),
            "label": list(ds.relation_to_id.keys()),
        }
    )

    df_all = pd.concat(
        [bundle.dfs["edges_train"], bundle.dfs["edges_test"], bundle.dfs["edges_val"]],
        ignore_index=True,
    )
    bundle.dfs["edges"] = pd.DataFrame(
        {
            "head": df_all["head"].tolist(),
            "tail": df_all["tail"].tolist(),
            "relation": df_all["relation"].tolist(),
        }
    )
    self.print(
        f"Dataset contains {len(bundle.dfs['nodes'])} nodes, ",
        f"{len(bundle.dfs['relations'])} relations and {len(bundle.dfs['edges'])} edges in total.",
    )
    return bundle

```
