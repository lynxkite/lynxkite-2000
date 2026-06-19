---
name: networkx-generators-joint-degree-seq
description: Collection of operations - Is valid joint degree, Is valid directed joint degree, Joint degree graph
---

**Is valid joint degree:**
Checks whether the given joint degree dictionary is realizable.

A *joint degree dictionary* is a dictionary of dictionaries, in
which entry ``joint_degrees[k][l]`` is an integer representing the
number of edges joining nodes of degree *k* with nodes of degree
*l*. Such a dictionary is realizable as a simple graph if and only
if the following conditions are satisfied.

- each entry must be an integer,
- the total number of nodes of degree *k*, computed by
  ``sum(joint_degrees[k].values()) / k``, must be an integer,
- the total number of edges joining nodes of degree *k* with
  nodes of degree *l* cannot exceed the total number of possible edges,
- each diagonal entry ``joint_degrees[k][k]`` must be even (this is
  a convention assumed by the :func:`joint_degree_graph` function).
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.joint_degree_seq.is_valid_joint_degree()

**Is valid directed joint degree:**
Checks whether the given directed joint degree input is realizable
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.joint_degree_seq.is_valid_directed_joint_degree()

**Joint degree graph:**
Generates a random simple graph with the given joint degree dictionary.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.joint_degree_seq.joint_degree_graph(seed=<seed_value>)
