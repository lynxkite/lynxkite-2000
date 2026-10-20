**Confusion matrix:**

```python
@op("Confusion matrix", view="matplotlib", color="blue", icon="dots-diagonal-2")
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
    fig, ax = plt.subplots(tight_layout=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xlabel("Predicted", labelpad=15)
    plt.ylabel("Actual", labelpad=15)
    plt.yticks(rotation=0)

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - prediction_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
