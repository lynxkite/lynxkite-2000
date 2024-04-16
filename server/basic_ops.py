'''Some operations. To be split into separate files when we have more.'''
from . import ops
import pandas as pd
import networkx as nx

@ops.op("Import Parquet")
def import_parquet(*, filename: str):
  '''Imports a parquet file.'''
  return pd.read_parquet(filename)

@ops.op("Create scale-free graph")
def create_scale_free_graph(*, nodes: int = 10):
  '''Creates a scale-free graph with the given number of nodes.'''
  return nx.scale_free_graph(nodes)

@ops.op("Compute PageRank")
def compute_pagerank(graph: nx.Graph, *, damping: 0.85, iterations: 3):
  return nx.pagerank(graph)

@ops.op("Visualize graph")
def visualize_graph(graph: ops.Bundle) -> 'graph_view':
  nodes = graph.dfs['nodes']['id'].tolist()
  edges = graph.dfs['edges'].drop_duplicates(['source', 'target'])
  edges = edges.to_dict(orient='records')
  return {
    'attributes': {},
    'options': {},
    'nodes': [{'key': id} for id in nodes],
    'edges': [{'key': str(r['source']) + ' -> ' + str(r['target']), **r} for r in edges],
  }

@ops.op("View table")
def view_table(dfs: ops.Bundle) -> 'table_view':
  v = {
    'dataframes': { name: {
      'columns': [str(c) for c in df.columns],
      'data': df.values.tolist(),
    } for name, df in dfs.dfs.items() },
    'edges': dfs.edges,
  }
  print(v)
  return v
