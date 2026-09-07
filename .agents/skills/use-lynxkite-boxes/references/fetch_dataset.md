**Fetch OpenML dataset:**
Fetches the specified dataset from OpenML.
```python
@op("Fetch OpenML dataset", slow=True, color="blue", icon="database-import")
def fetch_dataset(*, dataset_name: str, version: int = 1) -> core.Bundle:
    """
    Fetches the specified dataset from OpenML.
    :param dataset_name: The name of the dataset to fetch.
    :param version: The version of the dataset to fetch.
    """
    b = core.Bundle()
    dataset = fetch_openml(dataset_name, version=version, as_frame=True)
    b.dfs[dataset_name] = dataset.frame
    return b

```
