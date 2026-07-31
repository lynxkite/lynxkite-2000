**Train/test split:**
Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.
```python
@op("Train/test split")
def train_test_split(
    bundle: core.Bundle, *, table_name: core.TableName, test_ratio: float = 0.1, seed=1234
):
    """Splits a dataframe in the bundle into separate "_train" and "_test" dataframes."""
    df = bundle.dfs[table_name]
    test = df.sample(frac=test_ratio, random_state=seed).reset_index()
    train = df.drop(test.index).reset_index()
    bundle = bundle.copy()
    bundle.dfs[f"{table_name}_train"] = train
    bundle.dfs[f"{table_name}_test"] = test
    return bundle

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
