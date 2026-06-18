---
name: interval-graph
description: Generates an interval graph for a list of intervals given.
---

**Interval graph:**
Generates an interval graph for a list of intervals given.

In graph theory, an interval graph is an undirected graph formed from a set
of closed intervals on the real line, with a vertex for each interval
and an edge between vertices whose intervals intersect.
It is the intersection graph of the intervals.

More information can be found at:
https://en.wikipedia.org/wiki/Interval_graph
parameters:


usage:
output_variable = networkx.generators.interval_graph.interval_graph()
