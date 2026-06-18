---
name: networkx-generators-degree-seq
description: Collection of operations - Configuration model, Directed configuration model, Havel–Hakimi graph, Directed Havel–Hakimi graph, Random degree sequence graph
---

**Configuration model:**
Returns a random graph with the given degree sequence.

The configuration model generates a random pseudograph (graph with
parallel edges and self loops) by randomly assigning edges to
match the given degree sequence.
parameters:
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.degree_seq.configuration_model(seed=<seed_value>)

**Directed configuration model:**
Returns a directed_random graph with the given degree sequences.

The configuration model generates a random directed pseudograph
(graph with parallel edges and self loops) by randomly assigning
edges to match the given degree sequences.
parameters:
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.degree_seq.directed_configuration_model(seed=<seed_value>)

**Havel–Hakimi graph:**
Returns a simple graph with given degree sequence constructed
using the Havel-Hakimi algorithm.
parameters:


usage:
output_variable = networkx.generators.degree_seq.havel_hakimi_graph()

**Directed Havel–Hakimi graph:**
Returns a directed graph with the given degree sequences.
parameters:


usage:
output_variable = networkx.generators.degree_seq.directed_havel_hakimi_graph()

**Random degree sequence graph:**
Returns a simple random graph with the given degree sequence.

If the maximum degree $d_m$ in the sequence is $O(m^{1/4})$ then the
algorithm produces almost uniform random graphs in $O(m d_m)$ time
where $m$ is the number of edges.
parameters:
  - seed: int | None = None - .
  - tries: int | None = 10 - .

usage:
output_variable = networkx.generators.degree_seq.random_degree_sequence_graph(seed=<seed_value>, tries=<tries_value>)
