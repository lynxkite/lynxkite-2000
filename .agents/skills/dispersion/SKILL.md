---
name: dispersion
description: Calculate dispersion between `u` and `v` in `G`.
---

**Dispersion:**
Calculate dispersion between `u` and `v` in `G`.

A link between two actors (`u` and `v`) has a high dispersion when their
mutual ties (`s` and `t`) are not well connected with each other.
parameters:
  - normalized: <class 'bool'> = None -
  - alpha: <class 'float'> = 1.0 -
  - b: <class 'float'> = 0.0 -
  - c: <class 'float'> = 0.0 -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.dispersion.dispersion(normalized=<normalized_value>, alpha=<alpha_value>, b=<b_value>, c=<c_value>, G=<G_variable>)
