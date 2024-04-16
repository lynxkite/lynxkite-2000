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
  return {
    'attributes': {
      'name': 'My Graph'
    },
    'options': {
      'allowSelfLoops': True,
      'multi': False,
      'type': 'mixed'
    },
    'nodes': [
      {'key': 'Thomas'},
      {'key': 'Eric'}
    ],
    'edges': [
      {
        'key': 'T->E',
        'source': 'Thomas',
        'target': 'Eric',
      }
    ]
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
