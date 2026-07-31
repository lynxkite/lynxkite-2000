**Is distance regular:**
Returns True if the graph is distance regular, False otherwise.

A connected graph G is distance-regular if for any nodes x,y
and any integers i,j=0,1,...,d (where d is the graph
diameter), the number of vertices at distance i from x and
distance j from y depends only on i,j and the graph distance
between x and y, independently of the choice of x and y.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_regular.is_distance_regular(G=<G_variable>)
