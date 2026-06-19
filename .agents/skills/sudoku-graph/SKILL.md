---
name: sudoku-graph
description: Returns the n-Sudoku graph. The default value of n is 3.
---

**Sudoku graph:**
Returns the n-Sudoku graph. The default value of n is 3.

The n-Sudoku graph is a graph with n^4 vertices, corresponding to the
cells of an n^2 by n^2 grid. Two distinct vertices are adjacent if and
only if they belong to the same row, column, or n-by-n box.
parameters:
  - n: <class 'int'> = 3 --The order of the Sudoku graph, equal to the square root of the
number of rows. The default is 3.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.sudoku.sudoku_graph(n=<n_value>)
