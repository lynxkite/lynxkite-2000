**Intersection array:**
Returns the intersection array of a distance-regular graph.

Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
such that for any 2 vertices x,y in G at a distance i=d(x,y), there
are exactly c_i neighbors of y at a distance of i-1 from x and b_i
neighbors of y at a distance of i+1 from x.

A distance regular graph's intersection array is given by,
[b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_regular.intersection_array(G=<G_variable>)
