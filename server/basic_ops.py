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
@ops.nx_node_attribute_func('pagerank')
def compute_pagerank(graph: nx.Graph, *, damping=0.85, iterations=100):
  return nx.pagerank(graph, alpha=damping, max_iter=iterations)


def _map_color(value):
  cmap = matplotlib.cm.get_cmap('viridis')
  value = (value - value.min()) / (value.max() - value.min())
  rgba = cmap(value)
  return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgba[:, :3]]

@ops.op("Visualize graph", view="visualization")
def visualize_graph(graph: ops.Bundle, *, color_nodes_by: 'node_attribute' = None):
  nodes = graph.dfs['nodes'].copy()
  node_attributes = sorted(nodes.columns)
  if color_nodes_by:
    nodes['color'] = _map_color(nodes[color_nodes_by])
  nodes = nodes.to_records()
  edges = graph.dfs['edges'].drop_duplicates(['source', 'target'])
  edges = edges.to_records()
  pos = nx.spring_layout(graph.to_nx(), iterations=max(1, int(10000/len(nodes))))
  v = {
    'animationDuration': 500,
    'animationEasingUpdate': 'quinticInOut',
    'series': [
      {
        'type': 'graph',
        'roam': True,
        'lineStyle': {
          'color': 'gray',
          'curveness': 0.3,
        },
        'emphasis': {
          'focus': 'adjacency',
          'lineStyle': {
            'width': 10,
          }
        },
        'data': [
          {
            'id': str(n.id),
            'x': float(pos[n.id][0]), 'y': float(pos[n.id][1]),
            # Adjust node size to cover the same area no matter how many nodes there are.
            'symbolSize': 50 / len(nodes) ** 0.5,
            'itemStyle': {'color': n.color} if color_nodes_by else {},
          }
          for n in nodes],
        'links': [
          {'source': str(r.source), 'target': str(r.target)}
          for r in edges],
      },
    ],
  }
  return v

@ops.op("View tables", view="table_view")
def view_tables(bundle: ops.Bundle):
  v = {
    'dataframes': { name: {
      'columns': [str(c) for c in df.columns],
      'data': df.values.tolist(),
    } for name, df in bundle.dfs.items() },
    'relations': bundle.relations,
    'other': bundle.other,
  }
  return v
