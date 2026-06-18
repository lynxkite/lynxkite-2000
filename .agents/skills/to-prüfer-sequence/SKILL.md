---
name: to-prüfer-sequence
description: Returns the Prüfer sequence of the given tree.
---

**To Prüfer sequence:**
Returns the Prüfer sequence of the given tree.

A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
*n* - 1, inclusive. The tree corresponding to a given Prüfer
sequence can be recovered by repeatedly joining a node in the
sequence with a node with the smallest potential degree according to
the sequence.
parameters:
  - T: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.tree.coding.to_prufer_sequence(T=<T_variable>)
