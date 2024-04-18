'''Some operations. To be split into separate files when we have more.'''
from . import ops
import matplotlib
import networkx as nx
import pandas as pd

@ops.op("Import Parquet")
def import_parquet(*, filename: str):
  '''Imports a parquet file.'''
  return pd.read_parquet(filename)

@ops.op("Create scale-free graph")
def create_scale_free_graph(*, nodes: int = 10):
  '''Creates a scale-free graph with the given number of nodes.'''
  return nx.scale_free_graph(nodes)

@ops.op("Compute PageRank")
def compute_pagerank(graph: nx.Graph, *, damping=0.85, iterations=3):
  graph = graph.copy()
  pr = nx.pagerank(graph, alpha=damping, max_iter=iterations)
  nx.set_node_attributes(graph, pr, 'pagerank')
  return graph


def _map_color(value):
  cmap = matplotlib.cm.get_cmap('viridis')
  value = (value - value.min()) / (value.max() - value.min())
  rgba = cmap(value)
  return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgba[:, :3]]

@ops.op("Visualize graph")
def visualize_graph(graph: ops.Bundle, *, color_nodes_by: 'node_attribute' = None) -> 'graph_view':
  nodes = graph.dfs['nodes'].copy()
  node_attributes = sorted(nodes.columns)
  if color_nodes_by:
    nodes['color'] = _map_color(nodes[color_nodes_by])
  nodes = nodes.to_records()
  edges = graph.dfs['edges'].drop_duplicates(['source', 'target'])
  edges = edges.to_records()
  v = {
    'node_attributes': node_attributes,
    'attributes': {},
    'options': {},
    'nodes': [
      {
        'key': str(n.id),
        'attributes': {'color': n.color, 'size': 5} if color_nodes_by else {}
      }
      for n in nodes],
    'edges': [
      {'key': str(r.source) + ' -> ' + str(r.target), 'source': str(r.source), 'target': str(r.target)}
      for r in edges],
  }
  return v

@ops.op("View tables")
def view_tables(dfs: ops.Bundle) -> 'table_view':
  v = {
    'dataframes': { name: {
      'columns': [str(c) for c in df.columns],
      'data': df.values.tolist(),
    } for name, df in dfs.dfs.items() },
    'relations': dfs.relations,
  }
  return v
