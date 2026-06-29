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
