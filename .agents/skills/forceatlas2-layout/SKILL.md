---
name: forceatlas2-layout
description: Position nodes using the ForceAtlas2 force-directed layout algorithm.
---

**ForceAtlas2 layout:**
Position nodes using the ForceAtlas2 force-directed layout algorithm.

This function applies the ForceAtlas2 layout algorithm [1]_ to a NetworkX graph,
positioning the nodes in a way that visually represents the structure of the graph.
The algorithm uses physical simulation to minimize the energy of the system,
resulting in a more readable layout.
parameters:
  - max_iter: <class 'int'> = 100 -
  - jitter_tolerance: <class 'float'> = 1.0 -
  - scaling_ratio: <class 'float'> = 2.0 -
  - gravity: <class 'float'> = 1.0 -
  - distributed_action: <class 'bool'> = None -
  - strong_gravity: <class 'bool'> = None -
  - weight: str | None = None -
  - linlog: <class 'bool'> = None -
  - seed: int | None = None -
  - dim: <class 'int'> = 2 -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.drawing.layout.forceatlas2_layout(max_iter=<max_iter_value>, jitter_tolerance=<jitter_tolerance_value>, scaling_ratio=<scaling_ratio_value>, gravity=<gravity_value>, distributed_action=<distributed_action_value>, strong_gravity=<strong_gravity_value>, weight=<weight_value>, linlog=<linlog_value>, seed=<seed_value>, dim=<dim_value>, G=<G_variable>)
