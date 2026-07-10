**Connect nodes on attribute:**
Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

Parameters:
- source_table: Name of the first table
- source_id: ID column in the first table
- source_attribute: Attribute column in the first table used for matching
- target_table: Name of the second table
- target_id: ID column in the second table
- target_attribute: Attribute column in the second table used for matching
```python
@op("Connect nodes on attribute", icon="share")
def connect_nodes(
    b: core.Bundle,
    *,
    source_table: core.TableName,
    source_id: ColumnNameForSource,
    source_attribute: ColumnNameForSource,
    target_table: core.TableName,
    target_id: ColumnNameForTarget,
    target_attribute: ColumnNameForTarget,
) -> core.Bundle:
    """
    Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

    Parameters:
    - source_table: Name of the first table
    - source_id: ID column in the first table
    - source_attribute: Attribute column in the first table used for matching
    - target_table: Name of the second table
    - target_id: ID column in the second table
    - target_attribute: Attribute column in the second table used for matching
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()

    df1 = (b.dfs[source_table]).add_suffix("_src")
    df2 = (b.dfs[target_table]).add_suffix("_dst")
    source_key, target_key = source_id, target_id
    source_attribute, target_attribute, source_id, target_id = (
        source_attribute + "_src",
        target_attribute + "_dst",
        source_id + "_src",
        target_id + "_dst",
    )
    edges = pd.merge(df1, df2, left_on=source_attribute, right_on=target_attribute)

    if source_table == target_table:
        edges[[source_id, target_id]] = np.sort(edges[[source_id, target_id]], axis=1)
        edges = edges[edges[source_id] != edges[target_id]].drop_duplicates(
            subset=[source_id, target_id]
        )
    b.dfs["edges"] = edges

    b.relations.append(
        core.RelationDefinition(
            name="graph",
            df="edges",
            source_column=source_id,
            source_table=source_table,
            source_key=source_key,
            target_column=target_id,
            target_table=target_table,
            target_key=target_key,
        )
    )
    return b

```
Custom types:
  - source_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - source_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}]
  - source_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}]
  - target_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - target_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}]
  - target_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}]
