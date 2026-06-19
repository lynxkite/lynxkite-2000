---
name: networkx-readwrite-sparse6
description: Collection of operations - From sparse6 bytes, Read sparse6
---

**From sparse6 bytes:**
Read an undirected graph in sparse6 format from string.
parameters:
  - string: <class 'str'> = ? --Data in sparse6 format

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.sparse6.from_sparse6_bytes(string=<string_value>)

**Read sparse6:**
Read an undirected graph in sparse6 format from path.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.sparse6.read_sparse6()
