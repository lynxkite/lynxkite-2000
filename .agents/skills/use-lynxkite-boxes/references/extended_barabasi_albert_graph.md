**Extended Barabasi–Albert graph:**
Returns an extended Barabási–Albert model graph.

An extended Barabási–Albert model graph is a random graph constructed
using preferential attachment. The extended model allows new edges,
rewired edges or new nodes. Based on the probabilities $p$ and $q$
with $p + q < 1$, the growing behavior of the graph is determined as:

1) With $p$ probability, $m$ new edges are added to the graph,
starting from randomly chosen existing nodes and attached preferentially at the
other end.

2) With $q$ probability, $m$ existing edges are rewired
by randomly choosing an edge and rewiring one end to a preferentially chosen node.

3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph
with edges attached preferentially.

When $p = q = 0$, the model behaves just like the Barabási–Alber model.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m: <class 'int'> = ? --Number of edges with which a new node attaches to existing nodes
  - p: <class 'float'> = ? --Probability value for adding an edge between existing nodes. p + q < 1
  - q: <class 'float'> = ? --Probability value of rewiring of existing edges. p + q < 1
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.extended_barabasi_albert_graph(n=<n_value>, m=<m_value>, p=<p_value>, q=<q_value>, seed=<seed_value>)
