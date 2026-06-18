---
name: is-tournament
description: Returns True if and only if `G` is a tournament.
---

**Is tournament:**
Returns True if and only if `G` is a tournament.

A tournament is a directed graph, with neither self-loops nor
multi-edges, in which there is exactly one directed edge joining
each pair of distinct nodes.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tournament.is_tournament(G=<G_variable>)
