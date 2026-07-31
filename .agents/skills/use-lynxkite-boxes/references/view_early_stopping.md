**View early stopping metric:**

```python
@op("View early stopping metric", view="visualization", color="blue", icon="chart-line")
def view_early_stopping(bundle: core.Bundle):
    metric = bundle.dfs["early_stopper_metric"].early_stopper_metric.tolist()
    v = {
        "title": {"text": "Early Stopping Metric"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"data": metric, "type": "line"}],
    }
    return v

```
