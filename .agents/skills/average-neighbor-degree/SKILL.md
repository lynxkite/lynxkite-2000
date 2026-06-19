---
name: average-neighbor-degree
description: Returns the average degree of the neighborhood of each node.
---

**Average neighbor degree:**
Returns the average degree of the neighborhood of each node.

In an undirected graph, the neighborhood `N(i)` of node `i` contains the
nodes that are connected to `i` by an edge.

For directed graphs, `N(i)` is defined according to the parameter `source`:

    - if source is 'in', then `N(i)` consists of predecessors of node `i`.
    - if source is 'out', then `N(i)` consists of successors of node `i`.
    - if source is 'in+out', then `N(i)` is both predecessors and successors.

The average neighborhood degree of a node `i` is

.. math::

    k_{nn,i} = \frac{1}{|N(i)|} \sum_{j \in N(i)} k_j

where `N(i)` are the neighbors of node `i` and `k_j` is
the degree of node `j` which belongs to `N(i)`. For weighted
graphs, an analogous measure can be defined [1]_,

.. math::

    k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j

where `s_i` is the weighted degree of node `i`, `w_{ij}`
is the weight of the edge that links `i` and `j` and
`N(i)` are the neighbors of node `i`.
parameters:
  - source: str | None = out --Directed graphs only.
Use "in"- or "out"-neighbors of source node.
  - target: str | None = out --Directed graphs only.
Use "in"- or "out"-degree for target node.
  - weight: str | None = ? --The edge attribute that holds the numerical value used as a weight.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.assortativity.neighbor_degree.average_neighbor_degree(source=<source_value>, target=<target_value>, weight=<weight_value>, G=<G_variable>)
