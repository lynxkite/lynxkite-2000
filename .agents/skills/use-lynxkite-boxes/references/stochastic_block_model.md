**Stochastic block model:**
Returns a stochastic block model graph.

This model partitions the nodes in blocks of arbitrary sizes, and places
edges between pairs of nodes independently, with a probability that depends
on the blocks.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.community.stochastic_block_model(seed=<seed_value>)
