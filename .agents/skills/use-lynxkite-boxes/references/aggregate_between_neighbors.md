**Aggregate between neighbors:**
Depending on the direction, aggregates the specified columns nodes in one table to their neighbors in the other.
```python
@op("Aggregate between neighbors", icon="topology-star-3")
def aggregate_between_neighbors(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
    add_suffixes: bool,
    direction: Direction,
    aggregations: AggregationAdderBetweenNeighbors,
) -> core.Bundle:
    """
    Depending on the direction, aggregates the specified columns nodes in one table to their neighbors in the other.
    :param b: the bundle to operate on
    :param relation_name: the relation connecting the two tables
    :param add_suffixes: whether to add suffixes or not
    :param direction: whether to aggregate "To" or "From" the target table
    :param aggregations: the aggregations to perform, specified as a list of tuples (column_name, aggregation_function)
    """
    b = b.copy()
    relation = next(r for r in b.relations if r.name == relation_name)
    _suffix_check(add_suffixes, [funcs for _, funcs in aggregations])

    to_neighbor = direction == Direction.to_neighbor
    primary_pre = "target" if to_neighbor else "source"
    secondary_pre = "source" if to_neighbor else "target"

    primary_table = getattr(relation, f"{primary_pre}_table")
    primary_key = getattr(relation, f"{primary_pre}_key")
    primary_col = getattr(relation, f"{primary_pre}_column")
    secondary_table = getattr(relation, f"{secondary_pre}_table")
    secondary_key = getattr(relation, f"{secondary_pre}_key")
    secondary_col = getattr(relation, f"{secondary_pre}_column")

    cols = [col for col, _ in aggregations]
    secondary_df = b.dfs[secondary_table][[secondary_key] + cols].copy()
    merged = b.dfs[relation.df].merge(
        secondary_df, left_on=secondary_col, right_on=secondary_key, how="inner"
    )

    aggregated = merged.groupby(primary_col).agg(dict(aggregations))
    aggregated.columns = [
        f"{col}_{func}" if add_suffixes else col for col, func in aggregated.columns
    ]
    aggregated = aggregated.reset_index().rename(columns={primary_col: primary_key})

    primary_df = b.dfs[primary_table].copy()
    primary_df = primary_df[
        [col for col in primary_df.columns if col not in aggregated.columns or col == primary_key]
    ]

    b.dfs[primary_table] = primary_df.merge(aggregated, on=primary_key, how="left")
    return b

```
Custom types:
  - relation_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}]
  - aggregations: typing.Annotated[list[tuple[str, list[str]]], {'format': 'dropdown-multidropdown_relation_adder', 'direction_map': {'Aggregate to neighbor': 'source_table', 'Aggregate from neighbor': 'target_table'}, 'options2': ['sum', 'mean', 'median', 'min', 'max', 'prod', 'std', 'var', 'sem', 'skew', 'count', 'size', 'first', 'last']}]
