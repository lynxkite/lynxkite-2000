---
name: use-lynxkite-boxes
description: Use the boxes already defined in LynxKite.
---
## Available boxes
The following boxes are available for use in your workflows.
Each box corresponds to a specific operation or function that can be used to build your workflow.
For detailed information on each box, please refer to the individual box documentation in the references folder.

**Cross-entropy loss:**
Cross-entropy loss
for usage information, see references/no_op.md

**Drop first n:**
Drop first n
for usage information, see references/no_op.md

**Graph conv:**
Graph conv
for usage information, see references/no_op.md

**Heterogeneous graph conv:**
Heterogeneous graph conv
for usage information, see references/no_op.md

**Optimizer:**
Optimizer
for usage information, see references/no_op.md

**Output:**
Output
for usage information, see references/no_op.md

**Pick element by constant:**
Pick element by constant
for usage information, see references/no_op.md

**Pick element by index:**
Pick element by index
for usage information, see references/no_op.md

**Recurrent chain:**
Recurrent chain
for usage information, see references/no_op.md

**Repeat:**
Repeat
for usage information, see references/no_op.md

**Take first n:**
Take first n
for usage information, see references/no_op.md

**Triplet margin loss:**
Triplet margin loss
for usage information, see references/no_op.md

**View tables:**
View tables
for usage information, see references/view_tables.md

**Export to file:**
Exports a DataFrame to a file.
for usage information, see references/export_to_file.md

**Graph from OSM:**
Graph from OSM
for usage information, see references/import_osm.md

**Import CSV:**
Imports a CSV file.
for usage information, see references/import_csv.md

**Import file:**
Read the contents of the a file into a `Bundle`.
for usage information, see references/import_file.md

**Import GraphML:**
Imports a GraphML file.
for usage information, see references/import_graphml.md

**Import Parquet:**
Imports a Parquet file.
for usage information, see references/import_parquet.md

**Aggregate on neighbors:**
Aggregate on neighbors
for usage information, see references/aggregate_on_neighbors.md

**Connect nodes on attribute:**
Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.
for usage information, see references/connect_nodes.md

**Define edges:**
Define edges between node tables
for usage information, see references/define_edges.md

**Degree:**
Degree
for usage information, see references/degree.md

**Discard loop edges:**
Discard loop edges
for usage information, see references/discard_loop_edges.md

**Discard loop edges in relation:**
Discards loop edges in the specified relation.
for usage information, see references/discard_loop_edges_in_relation.md

**Discard parallel edges:**
Discard parallel edges
for usage information, see references/discard_parallel_edges.md

**Graph from edge list:**
Graph from edge list
for usage information, see references/graph_from_edge_list.md

**Merge:**
Merge multiple inputs
for usage information, see references/merge.md

**Merge nodes on attribute:**
Merges the nodes that have the same value for the given attribute.
for usage information, see references/merge_nodes.md

**Merge parallel edges:**
Merges parallel edges, and aggregates the attributes with the specified functions(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats).
for usage information, see references/merge_parallel_edges.md

**Sample graph:**
Takes a (preferably connected) subgraph.
for usage information, see references/sample_graph.md

**Define model:**
Trains the selected model on the selected dataset. Most training parameters are set in the model definition.
for usage information, see references/define_model.md

**Model inference:**
Executes a trained model.
for usage information, see references/model_inference.md

**Train model:**
Trains the selected model on the selected dataset.
for usage information, see references/train_model.md

**Train/test split:**
Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.
for usage information, see references/train_test_split.md

**Train/test/validation split:**
Splits a dataframe in the bundle into separate "_train", "_test" and "_val" dataframes.
for usage information, see references/train_test_val_split.md

**View loss:**
View loss
for usage information, see references/view_loss.md

**View vectors:**
View vectors
for usage information, see references/view_vectors.md

**Define inductive PyKEEN model:**
Defines an InductiveNodePiece model (with an optional GNN message passing layer) for inductive link prediction tasks.
for usage information, see references/get_inductive_model.md

**Define PyKEEN model:**
Defines a PyKEEN model based on the selected model type.
for usage information, see references/define_pykeen_model.md

**Define PyKEEN model with node attributes:**
Defines a PyKEEN model capable of using numeric literals as node attributes.
for usage information, see references/def_pykeen_with_attributes.md

**Evaluate inductive model:**
Evaluate inductive model
for usage information, see references/eval_inductive_model.md

**Evaluate model:**
Evaluates the given model on the test set using the specified evaluator type.
for usage information, see references/evaluate.md

**Extract embeddings from PyKEEN model:**
Extract embeddings from PyKEEN model
for usage information, see references/extract_from_pykeen.md

**Full prediction:**
Warning: This prediction can be a very expensive operation!
for usage information, see references/full_predict.md

**Import inductive dataset:**
Imports an inductive dataset from the PyKEEN library.
for usage information, see references/import_inductive_dataset.md

**Import PyKEEN dataset:**
Imports a dataset from the PyKEEN library.
for usage information, see references/import_pykeen_dataset_path.md

**Split inductive dataset:**
Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
for usage information, see references/inductively_split_dataset.md

**Target prediction:**
Leave the target prediction field empty
for usage information, see references/target_predict.md

**Train embedding model:**
Train embedding model
for usage information, see references/train_embedding_model.md

**Train inductive model:**
Train inductive model
for usage information, see references/train_inductive_pykeen_model.md

**Triples prediction:**
Triples prediction
for usage information, see references/triple_predict.md

**View early stopping metric:**
View early stopping metric
for usage information, see references/view_early_stopping.md

**Cypher:**
Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame.
for usage information, see references/cypher.md

**SQL:**
Run a SQL query on the DataFrames in the bundle. Save the results as a new DataFrame.
for usage information, see references/sql.md

**Aggregate from segmentation:**
For every node it aggregates the specified parameters of every node that share a segment with it.
for usage information, see references/aggregate_from_segmentation.md

**Aggregate to segmentation:**
For every segment in the segmentation it aggregates the specified parameters of the nodes belonging to it.
for usage information, see references/aggregate_to_segmentation.md

**Find connected components:**
Finds connected components in the graph of the relation.
for usage information, see references/connected_components.md

**Segment by attribute:**
Segments the nodes in a table based on the values of the specified attribute.
for usage information, see references/segment_by_attribute.md

**Add rank attribute:**
Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row
for usage information, see references/add_rank.md

**Derive property:**
Derive property
for usage information, see references/derive_property.md

**Enter table data:**
Enter table data as CSV. The first row should contain column names.
for usage information, see references/enter_table_data.md

**Filter tables:**
Keeps/removes the selected tables based on the value of drop_selected
for usage information, see references/filter_tables.md

**Filter with formula:**
Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false
for usage information, see references/filter_with_formula.md

**Join tables:**
Join/merge dataframes from two bundles.
for usage information, see references/join_tables.md

**Rename table:**
Assigns a new name to the table
for usage information, see references/rename_table.md

**Sample table:**
Sample table
for usage information, see references/sample_table.md

**Vector from attributes:**
Creates a new column with vectors that contain the selected attributes in the selected order
for usage information, see references/vector_from_attributes.md

**Bar chart:**
Bar chart
for usage information, see references/bar_chart.md

**Binned graph visualization:**
Nodes binned together by x and y are aggregated into one node.
for usage information, see references/binned_graph_visualization.md

**Histogram:**
Histogram
for usage information, see references/histogram.md

**Scatter plot:**
Scatter plot
for usage information, see references/scatter_plot.md

**Visualize graph:**
Visualize graph
for usage information, see references/visualize_graph.md

**Activation:**
Activation
for usage information, see references/activation.md

**Add:**
Add
for usage information, see references/<lambda>.md

**Attention:**
Attention
for usage information, see references/attention.md

**Binary cross-entropy with logits loss:**
Binary cross-entropy with logits loss
for usage information, see references/binary_cross_entropy_loss.md

**Concatenate:**
Concatenate
for usage information, see references/concatenate.md

**Constant vector:**
Constant vector
for usage information, see references/constant_vector.md

**Cos:**
Cos
for usage information, see references/<lambda>.md

**Dropout:**
Dropout
for usage information, see references/dropout.md

**Embedding:**
Embedding
for usage information, see references/embedding.md

**Exp:**
Exp
for usage information, see references/<lambda>.md

**Input: graph edges:**
The edges of a graph as input. A 2xE tensor of src/dst indices. Not batched.
for usage information, see references/graph_edges_input.md

**Input: sequential:**
An input tensor with a sequence for each sample.
for usage information, see references/sequential_input.md

**Input: tensor:**
An input tensor.
for usage information, see references/tensor_input.md

**LayerNorm:**
LayerNorm
for usage information, see references/layernorm.md

**Linear:**
Linear
for usage information, see references/linear.md

**Log:**
Log
for usage information, see references/<lambda>.md

**LSTM:**
LSTM
for usage information, see references/lstm.md

**Mean pool:**
Mean pool
for usage information, see references/mean_pool.md

**MSE loss:**
MSE loss
for usage information, see references/mse_loss.md

**Multiply:**
Multiply
for usage information, see references/<lambda>.md

**Neural ODE with MLP:**
A neural ODE for predicting a 1-dimensional value over time, using an MLP to model the derivative.
for usage information, see references/neural_ode_mlp.md

**Sin:**
Sin
for usage information, see references/<lambda>.md

**Softmax:**
Softmax
for usage information, see references/softmax.md

**Subtract:**
Subtract
for usage information, see references/<lambda>.md

**Blur:**
Blur
for usage information, see references/blur.md

**Crop:**
Crop
for usage information, see references/crop.md

**Detail:**
Detail
for usage information, see references/detail.md

**Edge enhance:**
Edge enhance
for usage information, see references/edge_enhance.md

**Flip horizontally:**
Flip horizontally
for usage information, see references/flip_horizontally.md

**Flip vertically:**
Flip vertically
for usage information, see references/flip_vertically.md

**Open image:**
Open image
for usage information, see references/open_image.md

**Save image:**
Save image
for usage information, see references/save_image.md

**To grayscale:**
To grayscale
for usage information, see references/to_grayscale.md

**View image:**
View image
for usage information, see references/view_image.md

**Average neighbor degree:**
Returns the average degree of the neighborhood of each node.
for usage information, see references/average_neighbor_degree.md

**Find asteroidal triple:**
Find an asteroidal triple in the given graph.
for usage information, see references/find_asteroidal_triple.md

**Is AT-free:**
Check if a graph is AT-free.
for usage information, see references/is_at_free.md

**Is bipartite:**
Returns True if graph G is bipartite, False if not.
for usage information, see references/is_bipartite.md

**Complete bipartite graph:**
Returns the complete bipartite graph `K_{n_1,n_2}`.
for usage information, see references/complete_bipartite_graph.md

**Local bridges:**
Iterate over local bridges of `G` optionally computing the span
for usage information, see references/local_bridges.md

**Tree broadcast center:**
Return the broadcast center of a tree.
for usage information, see references/tree_broadcast_center.md

**Tree broadcast time:**
Return the minimum broadcast time of a (node in a) tree.
for usage information, see references/tree_broadcast_time.md

**Betweenness centrality:**
Compute the shortest-path betweenness centrality for nodes.
for usage information, see references/betweenness_centrality.md

**Edge betweenness centrality:**
Compute betweenness centrality for edges.
for usage information, see references/edge_betweenness_centrality.md

**Closeness centrality:**
Compute closeness centrality for nodes.
for usage information, see references/closeness_centrality.md

**Degree centrality:**
Compute the degree centrality for nodes.
for usage information, see references/degree_centrality.md

**In degree centrality:**
Compute the in-degree centrality for nodes.
for usage information, see references/in_degree_centrality.md

**Out degree centrality:**
Compute the out-degree centrality for nodes.
for usage information, see references/out_degree_centrality.md

**Dispersion:**
Calculate dispersion between `u` and `v` in `G`.
for usage information, see references/dispersion.md

**Eigenvector centrality:**
Compute the eigenvector centrality for the graph G.
for usage information, see references/eigenvector_centrality.md

**Eigenvector centrality NumPy:**
Compute the eigenvector centrality for the graph `G`.
for usage information, see references/eigenvector_centrality_numpy.md

**Group betweenness centrality:**
Compute the group betweenness centrality for a group of nodes.
for usage information, see references/group_betweenness_centrality.md

**Prominent group:**
Find the prominent group of size $k$ in graph $G$. The prominence of the
for usage information, see references/prominent_group.md

**Katz centrality:**
Compute the Katz centrality for the nodes of the graph G.
for usage information, see references/katz_centrality.md

**Katz centrality NumPy:**
Compute the Katz centrality for the graph G.
for usage information, see references/katz_centrality_numpy.md

**Laplacian centrality:**
Compute the Laplacian centrality for nodes in the graph `G`.
for usage information, see references/laplacian_centrality.md

**Edge load centrality:**
Compute edge load.
for usage information, see references/edge_load_centrality.md

**Percolation centrality:**
Compute the percolation centrality for nodes.
for usage information, see references/percolation_centrality.md

**Global reaching centrality:**
Returns the global reaching centrality of a directed graph.
for usage information, see references/global_reaching_centrality.md

**Second order centrality:**
Compute the second order centrality for nodes of G.
for usage information, see references/second_order_centrality.md

**Communicability betweenness centrality:**
Returns subgraph communicability for all pairs of nodes in G.
for usage information, see references/communicability_betweenness_centrality.md

**Estrada index:**
Returns the Estrada index of a the graph G.
for usage information, see references/estrada_index.md

**Subgraph centrality:**
Returns subgraph centrality for each node in G.
for usage information, see references/subgraph_centrality.md

**Subgraph centrality exp:**
Returns the subgraph centrality for each node of G.
for usage information, see references/subgraph_centrality_exp.md

**Voterank:**
Select a list of influential nodes in a graph using VoteRank algorithm
for usage information, see references/voterank.md

**Chordal graph cliques:**
Returns all maximal cliques of a chordal graph.
for usage information, see references/chordal_graph_cliques.md

**Chordal graph treewidth:**
Returns the treewidth of the chordal graph G.
for usage information, see references/chordal_graph_treewidth.md

**Complete to chordal graph:**
Return a copy of G completed to a chordal graph
for usage information, see references/complete_to_chordal_graph.md

**Is chordal:**
Checks whether G is a chordal graph.
for usage information, see references/is_chordal.md

**Enumerate all cliques:**
Returns all cliques in an undirected graph.
for usage information, see references/enumerate_all_cliques.md

**Find cliques:**
Returns all maximal cliques in an undirected graph.
for usage information, see references/find_cliques.md

**Find cliques recursive:**
Returns all maximal cliques in a graph.
for usage information, see references/find_cliques_recursive.md

**Make max clique graph:**
Returns the maximal clique graph of the given graph.
for usage information, see references/make_max_clique_graph.md

**Max weight clique:**
Find a maximum weight clique in G.
for usage information, see references/max_weight_clique.md

**All triangles:**
Yields all unique triangles in an undirected graph.
for usage information, see references/all_triangles.md

**Average clustering:**
Compute the average clustering coefficient for the graph G.
for usage information, see references/average_clustering.md

**Clustering:**
Compute the clustering coefficient for nodes.
for usage information, see references/clustering.md

**Generalized degree:**
Compute the generalized degree for nodes.
for usage information, see references/generalized_degree.md

**Square clustering:**
Compute the squares clustering coefficient for nodes.
for usage information, see references/square_clustering.md

**Transitivity:**
Compute graph transitivity, the fraction of all possible triangles
for usage information, see references/transitivity.md

**Triangles:**
Compute the number of triangles.
for usage information, see references/triangles.md

**Equitable color:**
Provides an equitable coloring for nodes of `G`.
for usage information, see references/equitable_color.md

**Greedy color:**
Color a graph using various strategies of greedy graph coloring.
for usage information, see references/greedy_color.md

**Communicability:**
Returns communicability between all pairs of nodes in G.
for usage information, see references/communicability.md

**Communicability exp:**
Returns communicability between all pairs of nodes in G.
for usage information, see references/communicability_exp.md

**Attracting components:**
Generates the attracting components in `G`.
for usage information, see references/attracting_components.md

**Is attracting component:**
Returns True if `G` consists of a single attracting component.
for usage information, see references/is_attracting_component.md

**Number attracting components:**
Returns the number of attracting components in `G`.
for usage information, see references/number_attracting_components.md

**Articulation points:**
Yield the articulation points, or cut vertices, of a graph.
for usage information, see references/articulation_points.md

**Biconnected component edges:**
Returns a generator of lists of edges, one list for each biconnected
for usage information, see references/biconnected_component_edges.md

**Biconnected components:**
Returns a generator of sets of nodes, one set for each biconnected
for usage information, see references/biconnected_components.md

**Is biconnected:**
Returns True if the graph is biconnected, False otherwise.
for usage information, see references/is_biconnected.md

**Connected components:**
Generate connected components.
for usage information, see references/connected_components.md

**Is connected:**
Returns True if the graph is connected, False otherwise.
for usage information, see references/is_connected.md

**Node connected component:**
Returns the set of nodes in the component of graph containing node n.
for usage information, see references/node_connected_component.md

**Number connected components:**
Returns the number of connected components.
for usage information, see references/number_connected_components.md

**Is semiconnected:**
Returns True if the graph is semiconnected, False otherwise.
for usage information, see references/is_semiconnected.md

**Condensation:**
Returns the condensation of G.
for usage information, see references/condensation.md

**Is strongly connected:**
Test directed graph for strong connectivity.
for usage information, see references/is_strongly_connected.md

**Kosaraju strongly connected components:**
Generate nodes in strongly connected components of graph.
for usage information, see references/kosaraju_strongly_connected_components.md

**Number strongly connected components:**
Returns number of strongly connected components in graph.
for usage information, see references/number_strongly_connected_components.md

**Strongly connected components:**
Generate nodes in strongly connected components of graph.
for usage information, see references/strongly_connected_components.md

**Is weakly connected:**
Test directed graph for weak connectivity.
for usage information, see references/is_weakly_connected.md

**Number weakly connected components:**
Returns the number of weakly connected components in G.
for usage information, see references/number_weakly_connected_components.md

**Weakly connected components:**
Generate weakly connected components of G.
for usage information, see references/weakly_connected_components.md

**Is k edge connected:**
Tests to see if a graph is k-edge-connected.
for usage information, see references/is_k_edge_connected.md

**K edge components:**
Generates nodes in each maximal k-edge-connected component in G.
for usage information, see references/k_edge_components.md

**K edge subgraphs:**
Generates nodes in each maximal k-edge-connected subgraph in G.
for usage information, see references/k_edge_subgraphs.md

**Core number:**
Returns the core number for each node.
for usage information, see references/core_number.md

**k-core:**
Returns the k-core of G.
for usage information, see references/k_core.md

**k-corona:**
Returns the k-corona of G.
for usage information, see references/k_corona.md

**k-crust:**
Returns the k-crust of G.
for usage information, see references/k_crust.md

**k-shell:**
Returns the k-shell of G.
for usage information, see references/k_shell.md

**k-truss:**
Returns the k-truss of `G`.
for usage information, see references/k_truss.md

**Onion layers:**
Returns the layer of each vertex in an onion decomposition of the graph.
for usage information, see references/onion_layers.md

**Chordless cycles:**
Find simple chordless cycles of a graph.
for usage information, see references/chordless_cycles.md

**Cycle basis:**
Returns a list of cycles which form a basis for cycles of G.
for usage information, see references/cycle_basis.md

**Find cycle:**
Returns a cycle found via depth-first traversal.
for usage information, see references/find_cycle.md

**Girth:**
Returns the girth of the graph.
for usage information, see references/girth.md

**Minimum cycle basis:**
Returns a minimum weight cycle basis for G
for usage information, see references/minimum_cycle_basis.md

**Recursive simple cycles:**
Find simple cycles (elementary circuits) of a directed graph.
for usage information, see references/recursive_simple_cycles.md

**Simple cycles:**
Find simple cycles (elementary circuits) of a graph.
for usage information, see references/simple_cycles.md

**Find minimal d-separator:**
Returns a minimal d-separating set between `x` and `y` if possible
for usage information, see references/find_minimal_d_separator.md

**Is d-separator:**
Return whether node sets `x` and `y` are d-separated by `z`.
for usage information, see references/is_d_separator.md

**Is minimal d-separator:**
Determine if `z` is a minimal d-separator for `x` and `y`.
for usage information, see references/is_minimal_d_separator.md

**All topological sorts:**
Returns a generator of _all_ topological sorts of the directed graph G.
for usage information, see references/all_topological_sorts.md

**Ancestors:**
Returns all nodes having a path to `source` in `G`.
for usage information, see references/ancestors.md

**Antichains:**
Generates antichains from a directed acyclic graph (DAG).
for usage information, see references/antichains.md

**DAG longest path:**
Returns the longest path in a directed acyclic graph (DAG).
for usage information, see references/dag_longest_path.md

**DAG longest path length:**
Returns the longest path length in a DAG
for usage information, see references/dag_longest_path_length.md

**DAG to branching:**
Returns a branching representing all (overlapping) paths from
for usage information, see references/dag_to_branching.md

**Descendants:**
Returns all nodes reachable from `source` in `G`.
for usage information, see references/descendants.md

**Is aperiodic:**
Returns True if `G` is aperiodic.
for usage information, see references/is_aperiodic.md

**Is directed acyclic graph:**
Returns True if the graph `G` is a directed acyclic graph (DAG) or
for usage information, see references/is_directed_acyclic_graph.md

**Lexicographical topological sort:**
Generate the nodes in the unique lexicographical topological sort order.
for usage information, see references/lexicographical_topological_sort.md

**Topological generations:**
Stratifies a DAG into generations.
for usage information, see references/topological_generations.md

**Topological sort:**
Returns a generator of nodes in topologically sorted order.
for usage information, see references/topological_sort.md

**Transitive closure:**
Returns transitive closure of a graph
for usage information, see references/transitive_closure.md

**Transitive closure DAG:**
Returns the transitive closure of a directed acyclic graph.
for usage information, see references/transitive_closure_dag.md

**Transitive reduction:**
Returns transitive reduction of a directed graph
for usage information, see references/transitive_reduction.md

**Barycenter:**
Calculate barycenter of a connected graph, optionally with edge weights.
for usage information, see references/barycenter.md

**Center:**
Returns the center of the graph G.
for usage information, see references/center.md

**Diameter:**
Returns the diameter of the graph G.
for usage information, see references/diameter.md

**Eccentricity:**
Returns the eccentricity of nodes in G.
for usage information, see references/eccentricity.md

**Effective graph resistance:**
Returns the Effective graph resistance of G.
for usage information, see references/effective_graph_resistance.md

**Harmonic diameter:**
Returns the harmonic diameter of the graph G.
for usage information, see references/harmonic_diameter.md

**Kemeny constant:**
Returns the Kemeny constant of the given graph.
for usage information, see references/kemeny_constant.md

**Periphery:**
Returns the periphery of the graph G.
for usage information, see references/periphery.md

**Radius:**
Returns the radius of the graph G.
for usage information, see references/radius.md

**Resistance distance:**
Returns the resistance distance between pairs of nodes in graph G.
for usage information, see references/resistance_distance.md

**Intersection array:**
Returns the intersection array of a distance-regular graph.
for usage information, see references/intersection_array.md

**Is distance regular:**
Returns True if the graph is distance regular, False otherwise.
for usage information, see references/is_distance_regular.md

**Is strongly regular:**
Returns True if and only if the given graph is strongly
for usage information, see references/is_strongly_regular.md

**Connected dominating set:**
Returns a connected dominating set.
for usage information, see references/connected_dominating_set.md

**Global efficiency:**
Returns the average global efficiency of the graph.
for usage information, see references/global_efficiency.md

**Local efficiency:**
Returns the average local efficiency of the graph.
for usage information, see references/local_efficiency.md

**Eulerian circuit:**
Returns an iterator over the edges of an Eulerian circuit in `G`.
for usage information, see references/eulerian_circuit.md

**Eulerian path:**
Return an iterator over the edges of an Eulerian path in `G`.
for usage information, see references/eulerian_path.md

**Eulerize:**
Transforms a graph into an Eulerian graph.
for usage information, see references/eulerize.md

**Has Eulerian path:**
Return True iff `G` has an Eulerian path.
for usage information, see references/has_eulerian_path.md

**Is Eulerian:**
Returns True if and only if `G` is Eulerian.
for usage information, see references/is_eulerian.md

**Max flow min cost:**
Returns a maximum (s, t)-flow of minimum cost.
for usage information, see references/max_flow_min_cost.md

**Min cost flow:**
Returns a minimum cost flow satisfying all demands in digraph G.
for usage information, see references/min_cost_flow.md

**Min cost flow cost:**
Find the cost of a minimum cost flow satisfying all demands in digraph G.
for usage information, see references/min_cost_flow_cost.md

**Network simplex:**
Find a minimum cost flow satisfying all demands in digraph G.
for usage information, see references/network_simplex.md

**Weisfeiler–Lehman graph hash:**
Return Weisfeiler Lehman (WL) graph hash.
for usage information, see references/weisfeiler_lehman_graph_hash.md

**Weisfeiler–Lehman subgraph hashes:**
Return a dictionary of subgraph hashes by node.
for usage information, see references/weisfeiler_lehman_subgraph_hashes.md

**Flow hierarchy:**
Returns the flow hierarchy of a directed network.
for usage information, see references/flow_hierarchy.md

**Is KL connected:**
Returns True if and only if `G` is locally `(k, l)`-connected.
for usage information, see references/is_kl_connected.md

**KL connected subgraph:**
Returns the maximum locally `(k, l)`-connected subgraph of `G`.
for usage information, see references/kl_connected_subgraph.md

**Isolates:**
Iterator over isolates in the graph.
for usage information, see references/isolates.md

**Number of isolates:**
Returns the number of isolates in the graph.
for usage information, see references/number_of_isolates.md

**Could be isomorphic:**
Returns False if graphs are definitely not isomorphic.
for usage information, see references/could_be_isomorphic.md

**Fast could be isomorphic:**
Returns False if graphs are definitely not isomorphic.
for usage information, see references/fast_could_be_isomorphic.md

**Faster could be isomorphic:**
Returns False if graphs are definitely not isomorphic.
for usage information, see references/faster_could_be_isomorphic.md

**VF2++ all isomorphisms:**
Yields all the possible mappings between G1 and G2.
for usage information, see references/vf2pp_all_isomorphisms.md

**VF2++ is isomorphic:**
Examines whether G1 and G2 are isomorphic.
for usage information, see references/vf2pp_is_isomorphic.md

**VF2++ isomorphism:**
Return an isomorphic mapping between `G1` and `G2` if it exists.
for usage information, see references/vf2pp_isomorphism.md

**Hits:**
Returns HITS hubs and authorities values for nodes.
for usage information, see references/hits.md

**Google matrix:**
Returns the Google matrix of the graph.
for usage information, see references/google_matrix.md

**PageRank:**
Returns the PageRank of the nodes in the graph.
for usage information, see references/pagerank.md

**Adamic–Adar index:**
Compute the Adamic-Adar index of all node pairs in ebunch.
for usage information, see references/adamic_adar_index.md

**Cn Soundarajan–Hopcroft:**
Count the number of common neighbors of all node pairs in ebunch
for usage information, see references/cn_soundarajan_hopcroft.md

**Common neighbor centrality:**
Return the CCPA score for each pair of nodes.
for usage information, see references/common_neighbor_centrality.md

**Jaccard coefficient:**
Compute the Jaccard coefficient of all node pairs in ebunch.
for usage information, see references/jaccard_coefficient.md

**Preferential attachment:**
Compute the preferential attachment score of all node pairs in ebunch.
for usage information, see references/preferential_attachment.md

**Ra index Soundarajan–Hopcroft:**
Compute the resource allocation index of all node pairs in
for usage information, see references/ra_index_soundarajan_hopcroft.md

**Resource allocation index:**
Compute the resource allocation index of all node pairs in ebunch.
for usage information, see references/resource_allocation_index.md

**Within inter cluster:**
Compute the ratio of within- and inter-cluster common neighbors
for usage information, see references/within_inter_cluster.md

**All pairs lowest common ancestor:**
Return the lowest common ancestor of all pairs or the provided pairs
for usage information, see references/all_pairs_lowest_common_ancestor.md

**Tree all pairs lowest common ancestor:**
Yield the lowest common ancestor for sets of pairs in a tree.
for usage information, see references/tree_all_pairs_lowest_common_ancestor.md

**Is matching:**
Return True if ``matching`` is a valid matching of ``G``
for usage information, see references/is_matching.md

**Is maximal matching:**
Return True if ``matching`` is a maximal matching of ``G``
for usage information, see references/is_maximal_matching.md

**Is perfect matching:**
Return True if ``matching`` is a perfect matching for ``G``
for usage information, see references/is_perfect_matching.md

**Max weight matching:**
Compute a maximum-weighted matching of G.
for usage information, see references/max_weight_matching.md

**Maximal matching:**
Find a maximal matching in the graph.
for usage information, see references/maximal_matching.md

**Min weight matching:**
Compute a minimum-weight maximum-cardinality matching of `G`.
for usage information, see references/min_weight_matching.md

**Contracted nodes:**
Returns the graph that results from contracting `u` and `v`.
for usage information, see references/contracted_nodes.md

**Identified nodes:**
Returns the graph that results from contracting `u` and `v`.
for usage information, see references/contracted_nodes.md

**Moral graph:**
Return the Moral Graph
for usage information, see references/moral_graph.md

**Non randomness:**
Compute the non-randomness of a graph.
for usage information, see references/non_randomness.md

**Compose:**
Compose graph G with H by combining nodes and edges into a single graph.
for usage information, see references/compose.md

**Difference:**
Returns a new graph that contains the edges that exist in G but not in H.
for usage information, see references/difference.md

**Disjoint union:**
Combine graphs G and H. The nodes are assumed to be unique (disjoint).
for usage information, see references/disjoint_union.md

**Full join:**
Returns the full join of graphs G and H.
for usage information, see references/full_join.md

**Intersection:**
Returns a new graph that contains only the nodes and the edges that exist in
for usage information, see references/intersection.md

**Symmetric difference:**
Returns new graph with edges that exist in either G or H but not both.
for usage information, see references/symmetric_difference.md

**Union:**
Combine graphs G and H. The names of nodes must be unique.
for usage information, see references/union.md

**Cartesian product:**
Returns the Cartesian product of G and H.
for usage information, see references/cartesian_product.md

**Corona product:**
Returns the Corona product of G and H.
for usage information, see references/corona_product.md

**Lexicographic product:**
Returns the lexicographic product of G and H.
for usage information, see references/lexicographic_product.md

**Modular product:**
Returns the Modular product of G and H.
for usage information, see references/modular_product.md

**Power:**
Returns the specified power of a graph.
for usage information, see references/power.md

**Strong product:**
Returns the strong product of G and H.
for usage information, see references/strong_product.md

**Tensor product:**
Returns the tensor product of G and H.
for usage information, see references/tensor_product.md

**Complement:**
Returns the graph complement of G.
for usage information, see references/complement.md

**Reverse:**
Returns the reverse directed graph of G.
for usage information, see references/reverse.md

**Is perfect graph:**
Return True if G is a perfect graph, else False.
for usage information, see references/is_perfect_graph.md

**Check planarity:**
Check if a graph is planar and return a counterexample or an embedding.
for usage information, see references/check_planarity.md

**Is planar:**
Returns True if and only if `G` is planar.
for usage information, see references/is_planar.md

**Chromatic polynomial:**
Returns the chromatic polynomial of `G`
for usage information, see references/chromatic_polynomial.md

**Tutte polynomial:**
Returns the Tutte polynomial of `G`
for usage information, see references/tutte_polynomial.md

**Overall reciprocity:**
Compute the reciprocity for the whole graph.
for usage information, see references/overall_reciprocity.md

**Reciprocity:**
Compute the reciprocity in a directed graph.
for usage information, see references/reciprocity.md

**Is regular:**
Determines whether a graph is regular.
for usage information, see references/is_regular.md

**K factor:**
Compute a `k`-factor of a graph.
for usage information, see references/k_factor.md

**Rich club coefficient:**
Returns the rich-club coefficient of the graph `G`.
for usage information, see references/rich_club_coefficient.md

**Floyd–Warshall:**
Find all-pairs shortest path lengths using Floyd's algorithm.
for usage information, see references/floyd_warshall.md

**Floyd–Warshall NumPy:**
Find all-pairs shortest path lengths using Floyd's algorithm.
for usage information, see references/floyd_warshall_numpy.md

**Floyd–Warshall predecessor and distance:**
Find all-pairs shortest path lengths using Floyd's algorithm.
for usage information, see references/floyd_warshall_predecessor_and_distance.md

**All pairs all shortest paths:**
Compute all shortest paths between all nodes.
for usage information, see references/all_pairs_all_shortest_paths.md

**Average shortest path length:**
Returns the average shortest path length.
for usage information, see references/average_shortest_path_length.md

**Shortest path:**
Compute shortest paths in the graph.
for usage information, see references/shortest_path.md

**Shortest path length:**
Compute shortest path lengths in the graph.
for usage information, see references/shortest_path_length.md

**All pairs shortest path:**
Compute shortest paths between all nodes.
for usage information, see references/all_pairs_shortest_path.md

**All pairs shortest path length:**
Computes the shortest path lengths between all nodes in `G`.
for usage information, see references/all_pairs_shortest_path_length.md

**Bidirectional shortest path:**
Returns a list of nodes in a shortest path between source and target.
for usage information, see references/bidirectional_shortest_path.md

**Predecessor:**
Returns dict of predecessors for the path from source to all nodes in G.
for usage information, see references/predecessor.md

**Single source shortest path:**
Compute shortest path between source
for usage information, see references/single_source_shortest_path.md

**Single target shortest path:**
Compute shortest path to target from all nodes that reach target.
for usage information, see references/single_target_shortest_path.md

**All pairs Bellman–Ford path:**
Compute shortest paths between all nodes in a weighted graph.
for usage information, see references/all_pairs_bellman_ford_path.md

**All pairs Bellman–Ford path length:**
Compute shortest path lengths between all nodes in a weighted graph.
for usage information, see references/all_pairs_bellman_ford_path_length.md

**All pairs Dijkstra:**
Find shortest weighted paths and lengths between all nodes.
for usage information, see references/all_pairs_dijkstra.md

**All pairs Dijkstra path:**
Compute shortest paths between all nodes in a weighted graph.
for usage information, see references/all_pairs_dijkstra_path.md

**All pairs Dijkstra path length:**
Compute shortest path lengths between all nodes in a weighted graph.
for usage information, see references/all_pairs_dijkstra_path_length.md

**Bellman–Ford path length:**
Returns the shortest path length from source to target
for usage information, see references/bellman_ford_path_length.md

**Bellman–Ford predecessor and distance:**
Compute shortest path lengths and predecessors on shortest paths
for usage information, see references/bellman_ford_predecessor_and_distance.md

**Dijkstra path length:**
Returns the shortest weighted path length in G from source to target.
for usage information, see references/dijkstra_path_length.md

**Dijkstra predecessor and distance:**
Compute weighted shortest path length and predecessors.
for usage information, see references/dijkstra_predecessor_and_distance.md

**Find negative cycle:**
Returns a cycle with negative total weight if it exists.
for usage information, see references/find_negative_cycle.md

**Goldberg Radzik:**
Compute shortest path lengths and predecessors on shortest paths
for usage information, see references/goldberg_radzik.md

**Johnson:**
Uses Johnson's Algorithm to compute shortest paths.
for usage information, see references/johnson.md

**Multi source Dijkstra:**
Find shortest weighted paths and lengths from a given set of
for usage information, see references/multi_source_dijkstra.md

**Multi source Dijkstra path:**
Find shortest weighted paths in G from a given set of source
for usage information, see references/multi_source_dijkstra_path.md

**Multi source Dijkstra path length:**
Find shortest weighted path lengths in G from a given set of
for usage information, see references/multi_source_dijkstra_path_length.md

**Negative edge cycle:**
Returns True if there exists a negative edge cycle anywhere in G.
for usage information, see references/negative_edge_cycle.md

**Single source Bellman–Ford:**
Compute shortest paths and lengths in a weighted graph G.
for usage information, see references/single_source_bellman_ford.md

**Single source Bellman–Ford path length:**
Compute the shortest path length between source and all other
for usage information, see references/single_source_bellman_ford_path_length.md

**Single source Dijkstra:**
Find shortest weighted paths and lengths from a source node.
for usage information, see references/single_source_dijkstra.md

**Single source Dijkstra path length:**
Find shortest weighted path lengths in G from a source node.
for usage information, see references/single_source_dijkstra_path_length.md

**Generate random paths:**
Randomly generate `sample_size` paths of length `path_length`.
for usage information, see references/generate_random_paths.md

**Shortest simple paths:**
Returns
for usage information, see references/shortest_simple_paths.md

**Lattice reference:**
Latticize the given graph by swapping edges.
for usage information, see references/lattice_reference.md

**Omega:**
Returns the small-world coefficient (omega) of a graph
for usage information, see references/omega.md

**Random reference:**
Compute a random graph by swapping edges of a given graph.
for usage information, see references/random_reference.md

**Sigma:**
Returns the small-world coefficient (sigma) of the given graph.
for usage information, see references/sigma.md

**s-metric:**
Returns the s-metric [1]_ of graph.
for usage information, see references/s_metric.md

**Spanner:**
Returns a spanner of the given graph with the given stretch.
for usage information, see references/spanner.md

**Constraint:**
Returns the constraint on all nodes in the graph ``G``.
for usage information, see references/constraint.md

**Effective size:**
Returns the effective size of all nodes in the graph ``G``.
for usage information, see references/effective_size.md

**Dedensify:**
Compresses neighborhoods around high-degree nodes
for usage information, see references/dedensify.md

**Snap aggregation:**
Creates a summary graph based on attributes and connectivity.
for usage information, see references/snap_aggregation.md

**Connected double edge swap:**
Attempts the specified number of double-edge swaps in the graph `G`.
for usage information, see references/connected_double_edge_swap.md

**Directed edge swap:**
Swap three edges in a directed graph while keeping the node degrees fixed.
for usage information, see references/directed_edge_swap.md

**Double edge swap:**
Swap two edges in the graph while keeping the node degrees fixed.
for usage information, see references/double_edge_swap.md

**Is tournament:**
Returns True if and only if `G` is a tournament.
for usage information, see references/is_tournament.md

**BFS labeled edges:**
Iterate over edges in a breadth-first search (BFS) labeled by type.
for usage information, see references/bfs_labeled_edges.md

**BFS layers:**
Returns an iterator of all the layers in breadth-first search traversal.
for usage information, see references/bfs_layers.md

**Descendants at distance:**
Returns all nodes at a fixed `distance` from `source` in `G`.
for usage information, see references/descendants_at_distance.md

**Edge BFS:**
A directed, breadth-first-search of edges in `G`, beginning at `source`.
for usage information, see references/edge_bfs.md

**Edge DFS:**
A directed, depth-first-search of edges in `G`, beginning at `source`.
for usage information, see references/edge_dfs.md

**Maximum branching:**
Returns a maximum branching from G.
for usage information, see references/maximum_branching.md

**Maximum spanning arborescence:**
Returns a maximum spanning arborescence from G.
for usage information, see references/maximum_spanning_arborescence.md

**Minimum branching:**
Returns a minimum branching from G.
for usage information, see references/minimum_branching.md

**Minimum spanning arborescence:**
Returns a minimum spanning arborescence from G.
for usage information, see references/minimum_spanning_arborescence.md

**To Prüfer sequence:**
Returns the Prüfer sequence of the given tree.
for usage information, see references/to_prufer_sequence.md

**Junction tree:**
Returns a junction tree of a given graph.
for usage information, see references/junction_tree.md

**Maximum spanning edges:**
Generate edges in a maximum spanning forest of an undirected
for usage information, see references/maximum_spanning_edges.md

**Maximum spanning tree:**
Returns a maximum spanning tree or forest on an undirected graph `G`.
for usage information, see references/maximum_spanning_tree.md

**Minimum spanning edges:**
Generate edges in a minimum spanning forest of an undirected
for usage information, see references/minimum_spanning_edges.md

**Minimum spanning tree:**
Returns a minimum spanning tree or forest on an undirected graph `G`.
for usage information, see references/minimum_spanning_tree.md

**Partition spanning tree:**
Find a spanning tree while respecting a partition of edges.
for usage information, see references/partition_spanning_tree.md

**Random spanning tree:**
Sample a random spanning tree using the edges weights of `G`.
for usage information, see references/random_spanning_tree.md

**Is arborescence:**
Returns True if `G` is an arborescence.
for usage information, see references/is_arborescence.md

**Is branching:**
Returns True if `G` is a branching.
for usage information, see references/is_branching.md

**Is forest:**
Returns True if `G` is a forest.
for usage information, see references/is_forest.md

**Is tree:**
Returns True if `G` is a tree.
for usage information, see references/is_tree.md

**All triads:**
A generator of all possible triads in G.
for usage information, see references/all_triads.md

**Is triad:**
Returns True if the graph G is a triad, else False.
for usage information, see references/is_triad.md

**Triad type:**
Returns the sociological triad type for a triad.
for usage information, see references/triad_type.md

**Triads by type:**
Returns a list of all triads for each triad type in a directed graph.
for usage information, see references/triads_by_type.md

**Number of walks:**
Returns the number of walks connecting each pair of nodes in `G`
for usage information, see references/number_of_walks.md

**Gutman index:**
Returns the Gutman Index for the graph `G`.
for usage information, see references/gutman_index.md

**Hyper wiener index:**
Returns the Hyper-Wiener index of the graph `G`.
for usage information, see references/hyper_wiener_index.md

**Schultz index:**
Returns the Schultz Index (of the first kind) of `G`
for usage information, see references/schultz_index.md

**Wiener index:**
Returns the Wiener index of the given graph.
for usage information, see references/wiener_index.md

**Is empty:**
Returns True if `G` has no edges.
for usage information, see references/is_empty.md

**Is negatively weighted:**
Returns True if `G` has negatively weighted edges.
for usage information, see references/is_negatively_weighted.md

**Is weighted:**
Returns True if `G` has weighted edges.
for usage information, see references/is_weighted.md

**Set edge attributes:**
Sets edge attributes from a given value or dictionary of values.
for usage information, see references/set_edge_attributes.md

**Set node attributes:**
Sets node attributes from a given value or dictionary of values.
for usage information, see references/set_node_attributes.md

**From dict of dicts:**
Returns a graph from a dictionary of dictionaries.
for usage information, see references/from_dict_of_dicts.md

**From dict of lists:**
Returns a graph from a dictionary of lists.
for usage information, see references/from_dict_of_lists.md

**From edgelist:**
Returns a graph from a list of edges.
for usage information, see references/from_edgelist.md

**From NumPy array:**
Returns a graph from a 2D NumPy array.
for usage information, see references/from_numpy_array.md

**From Pandas adjacency:**
Returns a graph from Pandas DataFrame.
for usage information, see references/from_pandas_adjacency.md

**From Pandas edgelist:**
Returns a graph from Pandas DataFrame containing an edge list.
for usage information, see references/from_pandas_edgelist.md

**From SciPy sparse array:**
Creates a new graph from an adjacency matrix given as a SciPy sparse
for usage information, see references/from_scipy_sparse_array.md

**To NumPy array:**
Returns the graph adjacency matrix as a NumPy array.
for usage information, see references/to_numpy_array.md

**To Pandas edgelist:**
Returns the graph edge list as a Pandas DataFrame.
for usage information, see references/to_pandas_edgelist.md

**To SciPy sparse array:**
Returns the graph adjacency matrix as a SciPy sparse array.
for usage information, see references/to_scipy_sparse_array.md

**ForceAtlas2 layout:**
Position nodes using the ForceAtlas2 force-directed layout algorithm.
for usage information, see references/forceatlas2_layout.md

**Graph atlas:**
Returns graph number `i` from the Graph Atlas.
for usage information, see references/graph_atlas.md

**Graph atlas g:**
Returns the list of all graphs with up to seven nodes named in the
for usage information, see references/graph_atlas_g.md

**Balanced tree:**
Returns the perfectly balanced `r`-ary tree of height `h`.
for usage information, see references/balanced_tree.md

**Barbell graph:**
Returns the Barbell Graph: two complete graphs connected by a path.
for usage information, see references/barbell_graph.md

**Binomial tree:**
Returns the Binomial Tree of order n.
for usage information, see references/binomial_tree.md

**Circulant graph:**
Returns the circulant graph $Ci_n(x_1, x_2, ..., x_m)$ with $n$ nodes.
for usage information, see references/circulant_graph.md

**Circular ladder graph:**
Returns the circular ladder graph $CL_n$ of length n.
for usage information, see references/circular_ladder_graph.md

**Complete graph:**
Return the complete graph `K_n` with n nodes.
for usage information, see references/complete_graph.md

**Complete multipartite graph:**
Returns the complete multipartite graph with the specified subset sizes.
for usage information, see references/complete_multipartite_graph.md

**Cycle graph:**
Returns the cycle graph $C_n$ of cyclically connected nodes.
for usage information, see references/cycle_graph.md

**Dorogovtsev–Goltsev–Mendes graph:**
Returns the hierarchically constructed Dorogovtsev--Goltsev--Mendes graph.
for usage information, see references/dorogovtsev_goltsev_mendes_graph.md

**Empty graph:**
Returns the empty graph with n nodes and zero edges.
for usage information, see references/empty_graph.md

**Full r-ary tree:**
Creates a full r-ary tree of `n` nodes.
for usage information, see references/full_rary_tree.md

**Kneser graph:**
Returns the Kneser Graph with parameters `n` and `k`.
for usage information, see references/kneser_graph.md

**Ladder graph:**
Returns the Ladder graph of length n.
for usage information, see references/ladder_graph.md

**Lollipop graph:**
Returns the Lollipop Graph; ``K_m`` connected to ``P_n``.
for usage information, see references/lollipop_graph.md

**Null graph:**
Returns the Null graph with no nodes or edges.
for usage information, see references/null_graph.md

**Path graph:**
Returns the Path graph `P_n` of linearly connected nodes.
for usage information, see references/path_graph.md

**Star graph:**
Return a star graph.
for usage information, see references/star_graph.md

**Tadpole graph:**
Returns the (m,n)-tadpole graph; ``C_m`` connected to ``P_n``.
for usage information, see references/tadpole_graph.md

**Trivial graph:**
Return the Trivial graph with one node (with label 0) and no edges.
for usage information, see references/trivial_graph.md

**Turan graph:**
Return the Turan Graph
for usage information, see references/turan_graph.md

**Wheel graph:**
Return the wheel graph
for usage information, see references/wheel_graph.md

**Random cograph:**
Returns a random cograph with $2 ^ n$ nodes.
for usage information, see references/random_cograph.md

**Caveman graph:**
Returns a caveman graph of `l` cliques of size `k`.
for usage information, see references/caveman_graph.md

**Connected caveman graph:**
Returns a connected caveman graph of `l` cliques of size `k`.
for usage information, see references/connected_caveman_graph.md

**Gaussian random partition graph:**
Generate a Gaussian random partition graph.
for usage information, see references/gaussian_random_partition_graph.md

**LFR benchmark graph:**
Returns the LFR benchmark graph.
for usage information, see references/LFR_benchmark_graph.md

**Planted partition graph:**
Returns the planted l-partition graph.
for usage information, see references/planted_partition_graph.md

**Random partition graph:**
Returns the random partition graph with a partition of sizes.
for usage information, see references/random_partition_graph.md

**Relaxed caveman graph:**
Returns a relaxed caveman graph.
for usage information, see references/relaxed_caveman_graph.md

**Ring of cliques:**
Defines a "ring of cliques" graph.
for usage information, see references/ring_of_cliques.md

**Stochastic block model:**
Returns a stochastic block model graph.
for usage information, see references/stochastic_block_model.md

**Windmill graph:**
Generate a windmill graph.
for usage information, see references/windmill_graph.md

**Configuration model:**
Returns a random graph with the given degree sequence.
for usage information, see references/configuration_model.md

**Directed configuration model:**
Returns a directed_random graph with the given degree sequences.
for usage information, see references/directed_configuration_model.md

**Directed Havel–Hakimi graph:**
Returns a directed graph with the given degree sequences.
for usage information, see references/directed_havel_hakimi_graph.md

**Havel–Hakimi graph:**
Returns a simple graph with given degree sequence constructed
for usage information, see references/havel_hakimi_graph.md

**Random degree sequence graph:**
Returns a simple random graph with the given degree sequence.
for usage information, see references/random_degree_sequence_graph.md

**G(n,c) graph:**
Returns the growing network with copying (GNC) digraph with `n` nodes.
for usage information, see references/gnc_graph.md

**G(n,r) graph:**
Returns the growing network with redirection (GNR) digraph with `n`
for usage information, see references/gnr_graph.md

**Random k-out graph:**
Returns a random `k`-out graph with preferential attachment.
for usage information, see references/random_k_out_graph.md

**Scale-free graph:**
Returns a scale-free directed graph.
for usage information, see references/scale_free_graph.md

**Duplication divergence graph:**
Returns an undirected graph using the duplication-divergence model.
for usage information, see references/duplication_divergence_graph.md

**Partial duplication graph:**
Returns a random graph using the partial duplication model.
for usage information, see references/partial_duplication_graph.md

**Chordal cycle graph:**
Returns the chordal cycle graph on `p` nodes.
for usage information, see references/chordal_cycle_graph.md

**Is regular expander:**
Determines whether the graph G is a regular expander. [1]_
for usage information, see references/is_regular_expander.md

**Margulis–Gabber–Galil graph:**
Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.
for usage information, see references/margulis_gabber_galil_graph.md

**Maybe regular expander graph:**
Utility for creating a random regular expander.
for usage information, see references/maybe_regular_expander_graph.md

**Paley graph:**
Returns the Paley $\frac{(p-1)}{2}$ -regular graph on $p$ nodes.
for usage information, see references/paley_graph.md

**Random regular expander graph:**
Returns a random regular expander graph on $n$ nodes with degree $d$.
for usage information, see references/random_regular_expander_graph.md

**Geometric edges:**
Returns edge list of node pairs within `radius` of each other.
for usage information, see references/geometric_edges.md

**Navigable small-world graph:**
Returns a navigable small-world graph.
for usage information, see references/navigable_small_world_graph.md

**Random geometric graph:**
Returns a random geometric graph in the unit cube of dimensions `dim`.
for usage information, see references/random_geometric_graph.md

**Soft random geometric graph:**
Returns a soft random geometric graph in the unit cube.
for usage information, see references/soft_random_geometric_graph.md

**Thresholded random geometric graph:**
Returns a thresholded random geometric graph in the unit cube.
for usage information, see references/thresholded_random_geometric_graph.md

**H(k,n) Harary graph:**
Return the Harary graph with given node connectivity and node number.
for usage information, see references/hkn_harary_graph.md

**H(n,m) Harary graph:**
Return the Harary graph with given numbers of nodes and edges.
for usage information, see references/hnm_harary_graph.md

**Random Internet as graph:**
Generates a random undirected graph resembling the Internet AS network
for usage information, see references/random_internet_as_graph.md

**General random intersection graph:**
Returns a random intersection graph with independent probabilities
for usage information, see references/general_random_intersection_graph.md

**K random intersection graph:**
Returns a intersection graph with randomly chosen attribute sets for
for usage information, see references/k_random_intersection_graph.md

**Uniform random intersection graph:**
Returns a uniform random intersection graph.
for usage information, see references/uniform_random_intersection_graph.md

**Interval graph:**
Generates an interval graph for a list of intervals given.
for usage information, see references/interval_graph.md

**Is valid directed joint degree:**
Checks whether the given directed joint degree input is realizable
for usage information, see references/is_valid_directed_joint_degree.md

**Is valid joint degree:**
Checks whether the given joint degree dictionary is realizable.
for usage information, see references/is_valid_joint_degree.md

**Joint degree graph:**
Generates a random simple graph with the given joint degree dictionary.
for usage information, see references/joint_degree_graph.md

**Grid 2D graph:**
Returns the two-dimensional grid graph.
for usage information, see references/grid_2d_graph.md

**Grid graph:**
Returns the *n*-dimensional grid graph.
for usage information, see references/grid_graph.md

**Hexagonal lattice graph:**
Returns an `m` by `n` hexagonal lattice graph.
for usage information, see references/hexagonal_lattice_graph.md

**Hypercube graph:**
Returns the *n*-dimensional hypercube graph.
for usage information, see references/hypercube_graph.md

**Triangular lattice graph:**
Returns the $m$ by $n$ triangular lattice graph.
for usage information, see references/triangular_lattice_graph.md

**Inverse line graph:**
Returns the inverse line graph of graph G.
for usage information, see references/inverse_line_graph.md

**Line graph:**
Returns the line graph of the graph or digraph `G`.
for usage information, see references/line_graph.md

**Mycielski graph:**
Generator for the n_th Mycielski Graph.
for usage information, see references/mycielski_graph.md

**Mycielskian:**
Returns the Mycielskian of a simple, undirected graph G
for usage information, see references/mycielskian.md

**Nonisomorphic trees:**
Generate nonisomorphic trees of specified `order`.
for usage information, see references/nonisomorphic_trees.md

**Number of nonisomorphic trees:**
Returns the number of nonisomorphic trees of the specified `order`.
for usage information, see references/number_of_nonisomorphic_trees.md

**Random clustered graph:**
Generate a random graph with the given joint independent edge degree and
for usage information, see references/random_clustered_graph.md

**Barabasi–Albert graph:**
Returns a random graph using Barabási–Albert preferential attachment
for usage information, see references/barabasi_albert_graph.md

**Binomial graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
for usage information, see references/gnp_random_graph.md

**Connected Watts–Strogatz graph:**
Returns a connected Watts–Strogatz small-world graph.
for usage information, see references/connected_watts_strogatz_graph.md

**Dense G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.
for usage information, see references/dense_gnm_random_graph.md

**Dual Barabasi–Albert graph:**
Returns a random graph using dual Barabási–Albert preferential attachment
for usage information, see references/dual_barabasi_albert_graph.md

**Erdos–Renyi graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
for usage information, see references/gnp_random_graph.md

**Extended Barabasi–Albert graph:**
Returns an extended Barabási–Albert model graph.
for usage information, see references/extended_barabasi_albert_graph.md

**Fast G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
for usage information, see references/fast_gnp_random_graph.md

**G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.
for usage information, see references/gnm_random_graph.md

**G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
for usage information, see references/gnp_random_graph.md

**Newman–Watts–Strogatz graph:**
Returns a Newman–Watts–Strogatz small-world graph.
for usage information, see references/newman_watts_strogatz_graph.md

**Power-law cluster graph:**
Holme and Kim algorithm for growing graphs with powerlaw
for usage information, see references/powerlaw_cluster_graph.md

**Random lobster graph:**
Returns a random lobster graph.
for usage information, see references/random_lobster_graph.md

**Random power-law tree:**
Returns a tree with a power law degree distribution.
for usage information, see references/random_powerlaw_tree.md

**Random power-law tree sequence:**
Returns a degree sequence for a tree with a power law distribution.
for usage information, see references/random_powerlaw_tree_sequence.md

**Random regular graph:**
Returns a random $d$-regular graph on $n$ nodes.
for usage information, see references/random_regular_graph.md

**Random shell graph:**
Returns a random shell graph for the constructor given.
for usage information, see references/random_shell_graph.md

**Watts–Strogatz graph:**
Returns a Watts–Strogatz small-world graph.
for usage information, see references/watts_strogatz_graph.md

**Bull graph:**
Returns the Bull Graph
for usage information, see references/bull_graph.md

**Chvatal graph:**
Returns the Chvátal Graph
for usage information, see references/chvatal_graph.md

**Cubical graph:**
Returns the 3-regular Platonic Cubical Graph
for usage information, see references/cubical_graph.md

**Desargues graph:**
Returns the Desargues Graph
for usage information, see references/desargues_graph.md

**Diamond graph:**
Returns the Diamond graph
for usage information, see references/diamond_graph.md

**Dodecahedral graph:**
Returns the Platonic Dodecahedral graph.
for usage information, see references/dodecahedral_graph.md

**Frucht graph:**
Returns the Frucht Graph.
for usage information, see references/frucht_graph.md

**Generalized petersen graph:**
Returns the Generalized Petersen Graph GP(n,k).
for usage information, see references/generalized_petersen_graph.md

**Heawood graph:**
Returns the Heawood Graph, a (3,6) cage.
for usage information, see references/heawood_graph.md

**Hoffman singleton graph:**
Returns the Hoffman-Singleton Graph.
for usage information, see references/hoffman_singleton_graph.md

**House graph:**
Returns the House graph (square with triangle on top)
for usage information, see references/house_graph.md

**House x graph:**
Returns the House graph with a cross inside the house square.
for usage information, see references/house_x_graph.md

**Icosahedral graph:**
Returns the Platonic Icosahedral graph.
for usage information, see references/icosahedral_graph.md

**Krackhardt kite graph:**
Returns the Krackhardt Kite Social Network.
for usage information, see references/krackhardt_kite_graph.md

**Moebius–Kantor graph:**
Returns the Moebius-Kantor graph.
for usage information, see references/moebius_kantor_graph.md

**Octahedral graph:**
Returns the Platonic Octahedral graph.
for usage information, see references/octahedral_graph.md

**Pappus graph:**
Returns the Pappus graph.
for usage information, see references/pappus_graph.md

**Petersen graph:**
Returns the Petersen Graph.
for usage information, see references/petersen_graph.md

**Sedgewick maze graph:**
Return a small maze with a cycle.
for usage information, see references/sedgewick_maze_graph.md

**Tetrahedral graph:**
Returns the 3-regular Platonic Tetrahedral graph.
for usage information, see references/tetrahedral_graph.md

**Truncated cube graph:**
Returns the skeleton of the truncated cube.
for usage information, see references/truncated_cube_graph.md

**Truncated tetrahedron graph:**
Returns the skeleton of the truncated Platonic tetrahedron.
for usage information, see references/truncated_tetrahedron_graph.md

**Tutte graph:**
Returns the Tutte graph.
for usage information, see references/tutte_graph.md

**Davis Southern women graph:**
Returns Davis Southern women social network.
for usage information, see references/davis_southern_women_graph.md

**Florentine families graph:**
Returns Florentine families graph.
for usage information, see references/florentine_families_graph.md

**Karate club graph:**
Returns Zachary's Karate Club graph.
for usage information, see references/karate_club_graph.md

**Les miserables graph:**
Returns coappearance network of characters in the novel Les Miserables.
for usage information, see references/les_miserables_graph.md

**Spectral graph forge:**
Returns a random simple graph with spectrum resembling that of `G`
for usage information, see references/spectral_graph_forge.md

**Stochastic graph:**
Returns a right-stochastic representation of directed graph `G`.
for usage information, see references/stochastic_graph.md

**Sudoku graph:**
Returns the n-Sudoku graph. The default value of n is 3.
for usage information, see references/sudoku_graph.md

**Visibility graph:**
Return a Visibility Graph of an input Time Series.
for usage information, see references/visibility_graph.md

**Prefix tree:**
Creates a directed prefix tree from a list of paths.
for usage information, see references/prefix_tree.md

**Prefix tree recursive:**
Recursively creates a directed prefix tree from a list of paths.
for usage information, see references/prefix_tree_recursive.md

**Random labeled rooted forest:**
Returns a labeled rooted forest with `n` nodes.
for usage information, see references/random_labeled_rooted_forest.md

**Random labeled rooted tree:**
Returns a labeled rooted tree with `n` nodes.
for usage information, see references/random_labeled_rooted_tree.md

**Random labeled tree:**
Returns a labeled tree on `n` nodes chosen uniformly at random.
for usage information, see references/random_labeled_tree.md

**Random unlabeled rooted forest:**
Returns a forest or list of forests selected at random.
for usage information, see references/random_unlabeled_rooted_forest.md

**Random unlabeled rooted tree:**
Returns a number of unlabeled rooted trees uniformly at random
for usage information, see references/random_unlabeled_rooted_tree.md

**Random unlabeled tree:**
Returns a tree or list of trees chosen randomly.
for usage information, see references/random_unlabeled_tree.md

**Triad graph:**
Returns the triad graph with the given name.
for usage information, see references/triad_graph.md

**Algebraic connectivity:**
Returns the algebraic connectivity of an undirected graph.
for usage information, see references/algebraic_connectivity.md

**Fiedler vector:**
Returns the Fiedler vector of a connected undirected graph.
for usage information, see references/fiedler_vector.md

**Spectral bisection:**
Bisect the graph using the Fiedler vector.
for usage information, see references/spectral_bisection.md

**Spectral ordering:**
Compute the spectral_ordering of a graph.
for usage information, see references/spectral_ordering.md

**Attr matrix:**
Returns the attribute matrix using attributes from `G` as a numpy array.
for usage information, see references/attr_matrix.md

**Attr sparse matrix:**
Returns a SciPy sparse array using attributes from G.
for usage information, see references/attr_sparse_matrix.md

**Bethe–Hessian matrix:**
Returns the Bethe Hessian matrix of G.
for usage information, see references/bethe_hessian_matrix.md

**Adjacency matrix:**
Returns adjacency matrix of `G`.
for usage information, see references/adjacency_matrix.md

**Incidence matrix:**
Returns incidence matrix of G.
for usage information, see references/incidence_matrix.md

**Directed combinatorial Laplacian matrix:**
Return the directed combinatorial Laplacian matrix of G.
for usage information, see references/directed_combinatorial_laplacian_matrix.md

**Directed Laplacian matrix:**
Returns the directed Laplacian matrix of G.
for usage information, see references/directed_laplacian_matrix.md

**Laplacian matrix:**
Returns the Laplacian matrix of G.
for usage information, see references/laplacian_matrix.md

**Normalized Laplacian matrix:**
Returns the normalized Laplacian matrix of G.
for usage information, see references/normalized_laplacian_matrix.md

**Directed modularity matrix:**
Returns the directed modularity matrix of G.
for usage information, see references/directed_modularity_matrix.md

**Modularity matrix:**
Returns the modularity matrix of G.
for usage information, see references/modularity_matrix.md

**Adjacency spectrum:**
Returns eigenvalues of the adjacency matrix of G.
for usage information, see references/adjacency_spectrum.md

**Bethe–Hessian spectrum:**
Returns eigenvalues of the Bethe Hessian matrix of G.
for usage information, see references/bethe_hessian_spectrum.md

**Laplacian spectrum:**
Returns eigenvalues of the Laplacian of G
for usage information, see references/laplacian_spectrum.md

**Modularity spectrum:**
Returns eigenvalues of the modularity matrix of G.
for usage information, see references/modularity_spectrum.md

**Normalized Laplacian spectrum:**
Return eigenvalues of the normalized Laplacian of G
for usage information, see references/normalized_laplacian_spectrum.md

**Parse adjlist:**
Parse lines of a graph adjacency list representation.
for usage information, see references/parse_adjlist.md

**Parse edgelist:**
Parse lines of an edge list representation of a graph.
for usage information, see references/parse_edgelist.md

**Read edgelist:**
Read a graph from a list of edges.
for usage information, see references/read_edgelist.md

**Read weighted edgelist:**
Read a graph as list of edges with numeric weights.
for usage information, see references/read_weighted_edgelist.md

**Read gexf:**
Read graph in GEXF format from path.
for usage information, see references/read_gexf.md

**Parse GML:**
Parse GML graph from a string or iterable.
for usage information, see references/parse_gml.md

**Read GML:**
Read graph in GML format from `path`.
for usage information, see references/read_gml.md

**From graph6 bytes:**
Read a simple undirected graph in graph6 format from bytes.
for usage information, see references/from_graph6_bytes.md

**Read graph6:**
Read simple undirected graphs in graph6 format from path.
for usage information, see references/read_graph6.md

**Parse GraphML:**
Read graph in GraphML format from string.
for usage information, see references/parse_graphml.md

**Read GraphML:**
Read graph in GraphML format from path.
for usage information, see references/read_graphml.md

**Parse LEDA:**
Read graph in LEDA format from string or iterable.
for usage information, see references/parse_leda.md

**Parse multiline adjlist:**
Parse lines of a multiline adjacency list representation of a graph.
for usage information, see references/parse_multiline_adjlist.md

**Parse Pajek:**
Parse Pajek format graph from string or iterable.
for usage information, see references/parse_pajek.md

**From sparse6 bytes:**
Read an undirected graph in sparse6 format from string.
for usage information, see references/from_sparse6_bytes.md

**Read sparse6:**
Read an undirected graph in sparse6 format from path.
for usage information, see references/read_sparse6.md

**Convert node labels to integers:**
Returns a copy of the graph G with the nodes relabeled using
for usage information, see references/convert_node_labels_to_integers.md
