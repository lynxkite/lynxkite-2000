---
name: dispersion
description: Calculate dispersion between `u` and `v` in `G`.
---

**Dispersion:**
Calculate dispersion between `u` and `v` in `G`.

A link between two actors (`u` and `v`) has a high dispersion when their
mutual ties (`s` and `t`) are not well connected with each other.
parameters:
  - normalized: <class 'bool'> = ? --If True (default) normalize by the embeddedness of the nodes (u and v).
  - alpha: <class 'float'> = 1.0 --Parameters for the normalization procedure. When `normalized` is True,
the dispersion value is normalized by::

    result = ((dispersion + b) ** alpha) / (embeddedness + c)

as long as the denominator is nonzero.
  - b: <class 'float'> = 0.0 --Parameters for the normalization procedure. When `normalized` is True,
the dispersion value is normalized by::

    result = ((dispersion + b) ** alpha) / (embeddedness + c)

as long as the denominator is nonzero.
  - c: <class 'float'> = 0.0 --Parameters for the normalization procedure. When `normalized` is True,
the dispersion value is normalized by::

    result = ((dispersion + b) ** alpha) / (embeddedness + c)

as long as the denominator is nonzero.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.dispersion.dispersion(normalized=<normalized_value>, alpha=<alpha_value>, b=<b_value>, c=<c_value>, G=<G_variable>)
