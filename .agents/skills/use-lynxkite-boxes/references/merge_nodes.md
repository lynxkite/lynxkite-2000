**Merge nodes on attribute:**
Merges the nodes that have the same value for the given attribute.
The aggregations parameter is a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats)) that specifies
which other columns should be included in the new DataFrame and how to aggregate them.
```python
@op("Merge nodes on attribute", icon="affiliate")
def merge_nodes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute: core.ColumnNameByTableName,
    add_suffixes: bool = False,
    aggregations: core.AggregationAdderByTableName,
) -> core.Bundle:
    """Merges the nodes that have the same value for the given attribute.
    The aggregations parameter is a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats)) that specifies
    which other columns should be included in the new DataFrame and how to aggregate them.
    :param b: the bundle
    :param table_name: the name of the table
    :param attribute: the name of the attribute to merge on
    :param add_suffixes: whether to add suffixes to the aggregated columns
    :param aggregations: the aggregations to perform, specified as a list of tuples
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()

    relations = b.relations.copy()
    b.relations = []
    for r in relations:
        b.relations.append(r.copy())

    old_df = b.dfs[table_name].copy()
    agg_dict = {}
    name_dict = {}

    for column, funcs in aggregations:
        if column not in agg_dict:
            agg_dict[column] = []
        if len(funcs) > 1 and not add_suffixes:
            raise ValueError(
                "Adding suffixes is required when multiple aggregation functions are specified for a column."
            )
        for func in funcs:
            if func not in agg_dict[column]:
                agg_dict[column].append(func)
            name_dict[(column, func)] = f"{column}_{func}" if add_suffixes else column
    grouped_df = old_df.groupby(attribute).agg(agg_dict).replace({float("nan"): None})
    grouped_df.columns = [name_dict.get(col) for col in grouped_df.columns]
    b.dfs[table_name] = grouped_df.reset_index()
    update_relations(b, table_name, attribute, old_df.set_index(get_id(b, table_name))[attribute])
    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
  - aggregations: typing.Annotated[list[tuple[str, list[str]]], {'format': 'dropdown-multidropdown_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]', 'options2': ['sum', 'mean', 'median', 'min', 'max', 'prod', 'std', 'var', 'sem', 'skew', 'count', 'size', 'first', 'last']}]
