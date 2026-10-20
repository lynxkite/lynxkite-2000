**Merge parallel edges:**
Merges parallel edges, and aggregates the attributes with the specified functions(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats).
```python
@op("Merge parallel edges", icon="arrows-right")
def merge_parallel_edges(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    source_key: core.ColumnNameByTableName,
    target_key: core.ColumnNameByTableName,
    aggregations: core.AggregationAdderByTableName,
) -> core.Bundle:
    """
    Merges parallel edges, and aggregates the attributes with the specified functions(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats).
    :param b: the bundle
    :param table_name: the name of the table
    :param source_key: the name of the key in the source table
    :param target_key: the name of the key in the target table
    :param aggregations: the aggregations to perform, specified as a list of tuples
    """
    b = b.copy()
    edges = b.dfs[table_name].copy()
    group_cols = [source_key, target_key]
    agg_dict = {}

    for column, funcs in aggregations:
        func_list = [f for f in funcs if f]
        if func_list:
            if column not in agg_dict:
                agg_dict[column] = []
            for func in func_list:
                if func not in agg_dict[column]:
                    agg_dict[column].append(func)

    if agg_dict:
        merged = edges.groupby(group_cols).agg(agg_dict).replace({float("nan"): None}).reset_index()
        new_columns = []
        for col, func in merged.columns:
            if func == "":
                new_columns.append(col)
            else:
                new_columns.append(f"{col}_{func}")

        merged.columns = new_columns
    else:
        merged = edges.drop_duplicates(subset=group_cols).reset_index(drop=True)

    b.dfs[table_name] = merged
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - source_key: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - target_key: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - aggregations: typing.Annotated[list[tuple[str, list[str]]], {'format': 'dropdown-multidropdown_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]', 'options2': ['sum', 'mean', 'median', 'min', 'max', 'prod', 'std', 'var', 'sem', 'skew', 'count', 'size', 'first', 'last']}]
