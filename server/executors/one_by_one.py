from .. import ops
from .. import workspace
import fastapi
import json
import pandas as pd
import traceback
import inspect
import typing

class Context(ops.BaseConfig):
  '''Passed to operation functions as "_ctx" if they have such a parameter.'''
  node: workspace.WorkspaceNode
  last_result: typing.Any = None

class Output(ops.BaseConfig):
  '''Return this to send values to specific outputs of a node.'''
  output_handle: str
  value: dict


def df_to_list(df):
  return [dict(zip(df.columns, row)) for row in df.values]

def has_ctx(op):
  sig = inspect.signature(op.func)
  return '_ctx' in sig.parameters

CACHES = {}

def register(env: str, cache: bool = True):
  '''Registers the one-by-one executor.'''
  if cache:
    CACHES[env] = {}
    cache = CACHES[env]
  else:
    cache = None
  ops.EXECUTORS[env] = lambda ws: execute(ws, ops.CATALOGS[env], cache=cache)

def get_stages(ws, catalog):
  '''Inputs on top are batch inputs. We decompose the graph into a DAG of components along these edges.'''
  nodes = {n.id: n for n in ws.nodes}
  batch_inputs = {}
  inputs = {}
  for edge in ws.edges:
    inputs.setdefault(edge.target, []).append(edge.source)
    node = nodes[edge.target]
    op = catalog[node.data.title]
    i = op.inputs[edge.targetHandle]
    if i.position == 'top':
      batch_inputs.setdefault(edge.target, []).append(edge.source)
  stages = []
  for bt, bss in batch_inputs.items():
    upstream = set(bss)
    new = set(bss)
    while new:
      n = new.pop()
      for i in inputs.get(n, []):
        if i not in upstream:
          upstream.add(i)
          new.add(i)
    stages.append(upstream)
  stages.sort(key=lambda s: len(s))
  stages.append(set(nodes))
  return stages

EXECUTOR_OUTPUT_CACHE = {}

def execute(ws, catalog, cache=None):
  nodes = {n.id: n for n in ws.nodes}
  contexts = {n.id: Context(node=n) for n in ws.nodes}
  edges = {n.id: [] for n in ws.nodes}
  for e in ws.edges:
    edges[e.source].append(e)
  tasks = {}
  NO_INPUT = object() # Marker for initial tasks.
  for node in ws.nodes:
    node.data.error = None
    op = catalog[node.data.title]
    # Start tasks for nodes that have no inputs.
    if not op.inputs:
      tasks[node.id] = [NO_INPUT]
  batch_inputs = {}
  # Run the rest until we run out of tasks.
  for stage in get_stages(ws, catalog):
    next_stage = {}
    while tasks:
      n, ts = tasks.popitem()
      if n not in stage:
        next_stage.setdefault(n, []).extend(ts)
        continue
      node = nodes[n]
      data = node.data
      op = catalog[data.title]
      params = {**data.params}
      if has_ctx(op):
        params['_ctx'] = contexts[node.id]
      results = []
      for task in ts:
        try:
          inputs = [
            batch_inputs[(n, i.name)] if i.position == 'top' else task
            for i in op.inputs.values()]
          if cache:
            key = json.dumps(fastapi.encoders.jsonable_encoder((inputs, params)))
            if key not in cache:
              cache[key] = op.func(*inputs, **params)
            result = cache[key]
          else:
            result = op(*inputs, **params)
        except Exception as e:
          traceback.print_exc()
          data.error = str(e)
          break
        contexts[node.id].last_result = result
        # Returned lists and DataFrames are considered multiple tasks.
        if isinstance(result, pd.DataFrame):
          result = df_to_list(result)
        elif not isinstance(result, list):
          result = [result]
        results.extend(result)
      else: # Finished all tasks without errors.
        if op.type == 'visualization' or op.type == 'table_view':
          data.display = results[0]
        for edge in edges[node.id]:
          t = nodes[edge.target]
          op = catalog[t.data.title]
          i = op.inputs[edge.targetHandle]
          if i.position == 'top':
            batch_inputs.setdefault((edge.target, edge.targetHandle), []).extend(results)
          else:
            tasks.setdefault(edge.target, []).extend(results)
    tasks = next_stage
