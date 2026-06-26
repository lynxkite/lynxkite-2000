**Recursive simple cycles:**
Find simple cycles (elementary circuits) of a directed graph.

A `simple cycle`, or `elementary circuit`, is a closed path where
no node appears twice. Two elementary circuits are distinct if they
are not cyclic permutations of each other.

This version uses a recursive algorithm to build a list of cycles.
You should probably use the iterator version called simple_cycles().
Warning: This recursive version uses lots of RAM!
It appears in NetworkX for pedagogical value.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.recursive_simple_cycles(G=<G_variable>)
