---
name: connect-nodes-on-attribute
description: Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.
---

Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

Parameters:
- source_table: Name of the first table
- source_id: ID column in the first table
- source_attribute: Attribute column in the first table used for matching
- target_table: Name of the second table
- target_id: ID column in the second table
- target_attribute: Attribute column in the second table used for matching

parameters:
  - b: core.Bundle = None
  - source_table: core.TableName = None
  - source_id: ColumnNameForSource = None
  - source_attribute: ColumnNameForSource = None
  - target_table: core.TableName = None
  - target_id: ColumnNameForTarget = None
  - target_attribute: ColumnNameForTarget = None

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.connect_nodes(b=<b_variable>, source_table=<source_table_value>, source_id=<source_id_value>, source_attribute=<source_attribute_value>, target_table=<target_table_value>, target_id=<target_id_value>, target_attribute=<target_attribute_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
