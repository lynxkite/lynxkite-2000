**Scatter plot:**

```python
@op("Scatter plot", icon="chart-dots", color="blue", view="matplotlib")
def scatter_plot(b: core.Bundle, *, x: core.TableColumn, y: core.TableColumn):
    table_x, column_x = x
    table_y, column_y = y
    dx = b.dfs[table_x][column_x]
    dy = b.dfs[table_y][column_y]
    correlation = dx.corr(dy)
    plt.figure(figsize=(6, 6))
    sns.regplot(x=dx, y=dy)
    plt.title(f"Correlation: {correlation:.2f}")
    plt.xlabel(column_x)
    plt.ylabel(column_y)

```
Custom types:
  - x: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
  - y: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
