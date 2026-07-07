**Bar chart:**

```python
@op("Bar chart", icon="chart-bar", color="blue", view="matplotlib")
def bar_chart(
    b: core.Bundle,
    *,
    x: core.TableColumn,
    y: core.TableColumn,
):
    table_x, column_x = x
    table_y, column_y = y
    if table_x == table_y:
        df = b.dfs[table_x]
    else:
        df = b.dfs[table_x].merge(b.dfs[table_y], left_index=True, right_index=True)
    sorted_df = df.sort_values(column_x)
    dx = sorted_df[column_x]
    dy = sorted_df[column_y]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=dx, y=dy)
    plt.xlabel(column_x)
    plt.ylabel(column_y)

```
Custom types:
  - x: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
  - y: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
