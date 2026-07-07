**Histogram:**

```python
@op("Histogram", icon="chart-histogram", color="blue", view="matplotlib")
def histogram(b: core.Bundle, *, column: core.TableColumn, bins: int = 20):
    table, col = column
    data = b.dfs[table][col]
    plt.figure(figsize=(6, 6))
    sns.histplot(data, bins=bins)
    plt.xlabel(col)
    plt.ylabel("Count")

```
Custom types:
  - column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}]
