**Dorogovtsev–Goltsev–Mendes graph:**
Returns the hierarchically constructed Dorogovtsev--Goltsev--Mendes graph.

The Dorogovtsev--Goltsev--Mendes [1]_ procedure deterministically produces a
scale-free graph with ``3/2 * (3**(n-1) + 1)`` nodes
and ``3**n`` edges for a given `n`.

Note that `n` denotes the number of times the state transition is applied,
starting from the base graph with ``n = 0`` (no transitions), as in [2]_.
This is different from the parameter ``t = n - 1`` in [1]_.

.. plot::

    >>> nx.draw(nx.dorogovtsev_goltsev_mendes_graph(3))
parameters:
  - n: <class 'int'> = ? --The generation number.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.dorogovtsev_goltsev_mendes_graph(n=<n_value>)
