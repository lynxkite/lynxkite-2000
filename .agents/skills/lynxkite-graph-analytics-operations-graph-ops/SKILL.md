---
name: lynxkite-graph-analytics-operations-graph-ops
description: Collection of operations - Merge, Define Edges, Connect nodes on attribute, Discard loop edges, Discard parallel edges, Sample graph, Graph from edge list, Degree, Aggregate on neighbors
---

**Merge:**
Merge multiple inputs
parameters:
  - merge_mode: <enum 'BundleMergeMode'> = must be unique --?
  - bundles: list[lynxkite_graph_analytics.bundle.Bundle] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.merge(merge_mode=<merge_mode_value>, bundles=<bundles_variable>)

**Define Edges:**
Define edges between node tables
parameters:
  - relations: <class 'str'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.define_edges(relations=<relations_value>, b=<b_variable>)

**Connect nodes on attribute:**
Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

Parameters:
- source_table: Name of the first table
- source_id: ID column in the first table
- source_attribute: Attribute column in the first table used for matching
- target_table: Name of the second table
- target_id: ID column in the second table
- target_attribute: Attribute column in the second table used for matching
parameters:
  - source_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - source_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}] = ? --?
  - source_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}] = ? --?
  - target_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - target_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}] = ? --?
  - target_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.connect_nodes(source_table=<source_table_value>, source_id=<source_id_value>, source_attribute=<source_attribute_value>, target_table=<target_table_value>, target_id=<target_id_value>, target_attribute=<target_attribute_value>, b=<b_variable>)

**Discard loop edges:**

parameters:
  - graph: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.discard_loop_edges(graph=<graph_variable>)

**Discard parallel edges:**

parameters:
  - graph: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.discard_parallel_edges(graph=<graph_variable>)

**Sample graph:**
Takes a (preferably connected) subgraph.
parameters:
  - nodes: <class 'int'> = 100 --?
  - graph: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.sample_graph(nodes=<nodes_value>, graph=<graph_variable>)

**Graph from edge list:**

parameters:
  - source: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}] = ? --?
  - target: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}] = ? --?
  - df: <class 'pandas.core.frame.DataFrame'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.graph_from_edge_list(source=<source_value>, target=<target_value>, df=<df_variable>)

**Degree:**

parameters:
  - g: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.degree(g=<g_variable>)

**Aggregate on neighbors:**

parameters:
  - property: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}] = ? --?
  - aggregation: <enum 'AggregationMethod'> = ? --?
  - g: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.aggregate_on_neighbors(property=<property_value>, aggregation=<aggregation_value>, g=<g_variable>)
