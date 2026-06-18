---
name: edge-load-centrality
description: Compute edge load.
---

**Edge load centrality:**
Compute edge load.

WARNING: This concept of edge load has not been analysed
or discussed outside of NetworkX that we know of.
It is based loosely on load_centrality in the sense that
it counts the number of shortest paths which cross each edge.
This function is for demonstration and testing purposes.
parameters:
  - cutoff: bool | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.load.edge_load_centrality(cutoff=<cutoff_value>, G=<G_variable>)
