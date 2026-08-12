**Confusion matrix:**

```python
@op("Confusion matrix", view="visualization", color="blue", icon="dots-diagonal-2")
def conf_matrix(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    label_column: core.ColumnNameByTableName,
    prediction_column: core.ColumnNameByTableName,
):
    df = b.dfs[table_name]
    y_true = df[label_column].astype(str)
    y_pred = df[prediction_column].astype(str)
    classes = list(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    max_val = max(int(cm.max()), 1)
    data = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            item = {
                "value": [j, len(classes) - 1 - i, int(val)],
                "itemStyle": {"color": f"rgba(0, 100, 200, {0.1 + 0.9 * (val / max_val)})"},
            }
            data.append(item)

    return {
        "xAxis": {
            "position": "top",
            "data": classes,
            "name": "Predicted",
            "nameLocation": "middle",
            "nameGap": 50,
        },
        "yAxis": {
            "data": classes[::-1],
            "name": "Actual",
            "nameLocation": "middle",
            "nameGap": 60,
        },
        "visualMap": {"show": False},
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": True, "color": "#000000", "fontWeight": "bold", "fontSize": 32},
            }
        ],
    }

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - prediction_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
