**Segment by attribute:**
Segments the nodes based on the values of the specified attribute.
parameters:
  - attribute: <class 'str'> = ? --the attribute to segment by
  - segmentation_name: <class 'str'> = ? --the name of the segmentation
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.segment_by_attribute(attribute=<attribute_value>, segmentation_name=<segmentation_name_value>, b=<b_variable>)
