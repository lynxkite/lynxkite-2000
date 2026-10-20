**Transitivity:**
Compute graph transitivity, the fraction of all possible triangles
present in G.

Possible triangles are identified by the number of "triads"
(two edges with a shared vertex).

The transitivity is

.. math::

    T = 3\frac{\#triangles}{\#triads}.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.cluster.transitivity(G=<G_variable>)
