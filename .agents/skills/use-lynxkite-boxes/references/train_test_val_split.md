**Train/test/validation split:**
Splits a dataframe in the bundle into separate "_train", "_test" and "_val" dataframes.
```python
@op("Train/test/validation split")
def train_test_val_split(
    bundle: core.Bundle,
    *,
    table_name: core.TableName,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed=1234,
):
    """Splits a dataframe in the bundle into separate "_train", "_test" and "_val" dataframes."""
    df = bundle.dfs[table_name]
    test = df.sample(frac=test_ratio, random_state=seed)
    remaining = df.drop(test.index)
    val = remaining.sample(frac=val_ratio / (1 - test_ratio), random_state=seed + 1)

    train = remaining.drop(val.index).reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    bundle = bundle.copy()
    bundle.dfs[f"{table_name}_train"] = train
    bundle.dfs[f"{table_name}_test"] = test
    bundle.dfs[f"{table_name}_val"] = val
    return bundle

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
