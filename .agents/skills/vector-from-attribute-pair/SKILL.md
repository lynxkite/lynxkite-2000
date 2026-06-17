---
name: vector-from-attribute-pair
description: Creates a new column with vectors that contain the two attributes
---

Creates a new column with vectors that contain the two attributes

parameters:
  - b: core.Bundle = None
  - table_name: core.TableName = None
  - attribute1: core.ColumnNameByTableName = None
  - attribute2: core.ColumnNameByTableName = None
  - new_name: str = None

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.vector_from_attribute_pair(b=<b_variable>, table_name=<table_name_value>, attribute1=<attribute1_value>, attribute2=<attribute2_value>, new_name=<new_name_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
