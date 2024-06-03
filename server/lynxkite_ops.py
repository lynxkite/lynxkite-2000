'''Some operations. To be split into separate files when we have more.'''
from . import ops
import matplotlib
import networkx as nx
import pandas as pd
import traceback

op = ops.op_registration('LynxKite')

@op("Import Parquet")
def import_parquet(*, filename: str):
  '''Imports a parquet file.'''
  return pd.read_parquet(filename)

@op("Create scale-free graph")
def create_scale_free_graph(*, nodes: int = 10):
  '''Creates a scale-free graph with the given number of nodes.'''
  return nx.scale_free_graph(nodes)

@op("Compute PageRank")
@ops.nx_node_attribute_func('pagerank')
def compute_pagerank(graph: nx.Graph, *, damping=0.85, iterations=100):
  return nx.pagerank(graph, alpha=damping, max_iter=iterations)


@op("Sample graph")
def sample_graph(graph: nx.Graph, *, nodes: int = 100):
  '''Takes a subgraph.'''
  return nx.scale_free_graph(nodes)


def _map_color(value):
  cmap = matplotlib.cm.get_cmap('viridis')
  value = (value - value.min()) / (value.max() - value.min())
  rgba = cmap(value)
  return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgba[:, :3]]

@op("Visualize graph", view="visualization")
def visualize_graph(graph: ops.Bundle, *, color_nodes_by: 'node_attribute' = None):
  nodes = graph.dfs['nodes'].copy()
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

@op("View tables", view="table_view")
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

@ops.register_executor('LynxKite')
def execute(ws):
    catalog = ops.CATALOGS['LynxKite']
    # Nodes are responsible for interpreting/executing their child nodes.
    nodes = [n for n in ws.nodes if not n.parentId]
    children = {}
    for n in ws.nodes:
        if n.parentId:
            children.setdefault(n.parentId, []).append(n)
    outputs = {}
    failed = 0
    while len(outputs) + failed < len(nodes):
        for node in nodes:
            if node.id in outputs:
                continue
            inputs = [edge.source for edge in ws.edges if edge.target == node.id]
            if all(input in outputs for input in inputs):
                inputs = [outputs[input] for input in inputs]
                data = node.data
                op = catalog[data.title]
                params = {**data.params}
                if op.sub_nodes:
                    sub_nodes = children.get(node.id, [])
                    sub_node_ids = [node.id for node in sub_nodes]
                    sub_edges = [edge for edge in ws.edges if edge.source in sub_node_ids]
                    params['sub_flow'] = {'nodes': sub_nodes, 'edges': sub_edges}
                try:
                  output = op(*inputs, **params)
                except Exception as e:
                  traceback.print_exc()
                  data.error = str(e)
                  failed += 1
                  continue
                if len(op.inputs) == 1 and op.inputs.get('multi') == '*':
                    # It's a flexible input. Create n+1 handles.
                    data.inputs = {f'input{i}': None for i in range(len(inputs) + 1)}
                data.error = None
                outputs[node.id] = output
                if op.type == 'visualization' or op.type == 'table_view':
                    data.view = output
