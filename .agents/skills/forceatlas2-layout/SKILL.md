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
  - max_iter: <class 'int'> = 100 --Number of iterations for the layout optimization.
  - jitter_tolerance: <class 'float'> = 1.0 --Controls the tolerance for adjusting the speed of layout generation.
  - scaling_ratio: <class 'float'> = 2.0 --Determines the scaling of attraction and repulsion forces.
  - gravity: <class 'float'> = 1.0 --Determines the amount of attraction on nodes to the center. Prevents islands
(i.e. weakly connected or disconnected parts of the graph)
from drifting away.
  - distributed_action: <class 'bool'> = ? --Distributes the attraction force evenly among nodes.
  - strong_gravity: <class 'bool'> = ? --Applies a strong gravitational pull towards the center.
  - weight: str | None = ? --The edge attribute that holds the numerical value used for
the edge weight. If None, then all edge weights are 1.
  - linlog: <class 'bool'> = ? --Uses logarithmic attraction instead of linear.
  - seed: int | None = ? --Used only for the initial positions in the algorithm.
Set the random state for deterministic node layouts.
If int, `seed` is the seed used by the random number generator,
if numpy.random.RandomState instance, `seed` is the random
number generator,
if None, the random number generator is the RandomState instance used
by numpy.random.
  - dim: <class 'int'> = 2 --Sets the dimensions for the layout. Ignored if `pos` is provided.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph to be laid out.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.drawing.layout.forceatlas2_layout(max_iter=<max_iter_value>, jitter_tolerance=<jitter_tolerance_value>, scaling_ratio=<scaling_ratio_value>, gravity=<gravity_value>, distributed_action=<distributed_action_value>, strong_gravity=<strong_gravity_value>, weight=<weight_value>, linlog=<linlog_value>, seed=<seed_value>, dim=<dim_value>, G=<G_variable>)
