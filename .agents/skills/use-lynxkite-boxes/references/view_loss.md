**View loss:**

```python
@op("View loss", view="visualization", icon="trending-down-3")
def view_loss(bundle: core.Bundle):
    loss = bundle.dfs["training"].training_loss.tolist()
    v = {
        "title": {"text": "Training loss"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"data": loss, "type": "line"}],
    }
    return v

```
