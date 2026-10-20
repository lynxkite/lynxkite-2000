**Segment by attribute:**
Segments the nodes in a table based on the values of the specified attribute.
Creates a new table with segmentation IDs, edge table connecting nodes to segments,
and a relation accordingly.
```python
@op("Segment by attribute", icon="category-2")
def segment_by_attribute(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute: core.ColumnNameByTableName,
    segmentation_name: str,
):
    """
    Segments the nodes in a table based on the values of the specified attribute.
    Creates a new table with segmentation IDs, edge table connecting nodes to segments,
    and a relation accordingly.
    :param b: the bundle
    :param table_name: the name of the table to segment
    :param attribute: the attribute to segment by
    :param segmentation_name: the name of the segmentation
    """
    b = b.copy()
    id_column = get_id(b, table_name)

    node_df = b.dfs[table_name]
    unique_values = node_df[attribute].unique()

    b.dfs[segmentation_name] = pd.DataFrame(
        {
            "id": range(len(unique_values)),
            attribute: unique_values,
        }
    )

    edge_table_name = f"{segmentation_name}_edges"
    b.dfs[edge_table_name] = pd.DataFrame(
        {
            "node_id": node_df[id_column],
            "segment_id": node_df[attribute].map({v: i for i, v in enumerate(unique_values)}),
        }
    )

    b.relations.append(
        core.RelationDefinition(
            name=f"{table_name}_{segmentation_name}",
            df=edge_table_name,
            source_column="node_id",
            target_column="segment_id",
            source_table=table_name,
            target_table=segmentation_name,
            source_key=id_column,
            target_key="id",
        )
    )

    return b

```
Custom types:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}]
  - attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}]
