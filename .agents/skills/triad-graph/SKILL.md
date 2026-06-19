---
name: triad-graph
description: Returns the triad graph with the given name.
---

**Triad graph:**
Returns the triad graph with the given name.

Each string in the following tuple is a valid triad name::

    (
        "003",
        "012",
        "102",
        "021D",
        "021U",
        "021C",
        "111D",
        "111U",
        "030T",
        "030C",
        "201",
        "120D",
        "120U",
        "120C",
        "210",
        "300",
    )

Each triad name corresponds to one of the possible valid digraph on
three nodes.
parameters:
  - triad_name: <class 'str'> = ? --The name of a triad, as described above.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.triads.triad_graph(triad_name=<triad_name_value>)
