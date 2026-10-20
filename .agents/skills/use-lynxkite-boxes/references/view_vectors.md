**View vectors:**

```python
@op("View vectors", view="visualization", color="blue")
def view_vectors(
    bundle: core.Bundle,
    *,
    table_name: core.TableName = "nodes",
    vector_column: core.ColumnNameByTableName = "",
    label_column: core.ColumnNameByTableName = "",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: UMAPMetric = UMAPMetric.euclidean,
):
    try:
        from cuml.manifold.umap import UMAP  # ty: ignore[unresolved-import]
    except ImportError:
        from umap import UMAP
    vec = np.stack(bundle.dfs[table_name][vector_column].to_numpy())
    umap = functools.partial(
        UMAP,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric.value,
    )
    proj = umap(n_components=2).fit_transform(vec)
    color = umap(n_components=1).fit_transform(vec)
    data = [[*p.tolist(), "", c.item()] for p, c in zip(proj, color)]
    if label_column:
        for i, row in enumerate(bundle.dfs[table_name][label_column]):
            data[i][2] = row
    size = 100 / len(data) ** 0.4
    v = {
        "title": {
            "text": f"UMAP projection of {vector_column}",
        },
        "visualMap": {
            "min": color[:, 0].min().item(),
            "max": color[:, 0].max().item(),
            "right": 10,
            "top": "center",
            "calculable": True,
            "dimension": 3,
            "inRange": {"color": VIRIDIS},
        },
        "tooltip": {"trigger": "item", "formatter": "GET_THIRD_VALUE"}
        if label_column
        else {"show": False},
        "xAxis": [{"type": "value"}],
        "yAxis": [{"type": "value"}],
        "series": [{"type": "scatter", "symbolSize": size, "data": data}],
    }
    return v

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - vector_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
