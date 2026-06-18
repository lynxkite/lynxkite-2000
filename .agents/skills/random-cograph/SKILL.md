---
name: random-cograph
description: Returns a random cograph with $2 ^ n$ nodes.
---

**Random cograph:**
Returns a random cograph with $2 ^ n$ nodes.

A cograph is a graph containing no path on four vertices.
Cographs or $P_4$-free graphs can be obtained from a single vertex
by disjoint union and complementation operations.

This generator starts off from a single vertex and performs disjoint
union and full join operations on itself.
The decision on which operation will take place is random.
parameters:
  - n: <class 'int'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.cographs.random_cograph(n=<n_value>, seed=<seed_value>)
