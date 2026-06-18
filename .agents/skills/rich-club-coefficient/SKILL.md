---
name: rich-club-coefficient
description: Returns the rich-club coefficient of the graph `G`.
---

**Rich club coefficient:**
Returns the rich-club coefficient of the graph `G`.

For each degree *k*, the *rich-club coefficient* is the ratio of the
number of actual to the number of potential edges for nodes with
degree greater than *k*:

.. math::

    \phi(k) = \frac{2 E_k}{N_k (N_k - 1)}

where `N_k` is the number of nodes with degree larger than *k*, and
`E_k` is the number of edges among those nodes.
parameters:
  - normalized: <class 'bool'> = None -
  - Q: <class 'float'> = 100 -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.richclub.rich_club_coefficient(normalized=<normalized_value>, Q=<Q_value>, seed=<seed_value>, G=<G_variable>)
