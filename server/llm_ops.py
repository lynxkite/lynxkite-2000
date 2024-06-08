'''For specifying an LLM agent logic flow.'''
from . import ops
import chromadb
import fastapi.encoders
import inspect
import jinja2
import json
import openai
import pandas as pd
import traceback
import typing
from . import workspace

client = openai.OpenAI(base_url="http://localhost:11434/v1")
jinja = jinja2.Environment()
chroma_client = chromadb.Client()
LLM_CACHE = {}
ENV = 'LLM logic'
op = ops.op_registration(ENV)

class Context(ops.BaseConfig):
  '''Passed to operation functions as "_ctx" if they have such a parameter.'''
  node: workspace.WorkspaceNode
  last_result: typing.Any = None

class Output(ops.BaseConfig):
  '''Return this to send values to specific outputs of a node.'''
  output_handle: str
  value: dict

def chat(*args, **kwargs):
  key = json.dumps({'args': args, 'kwargs': kwargs})
  if key not in LLM_CACHE:
    completion = client.chat.completions.create(*args, **kwargs)
    LLM_CACHE[key] = [c.message.content for c in completion.choices]
  return LLM_CACHE[key]

@op("Input")
def input(*, filename: ops.PathStr, key: str):
  return pd.read_csv(filename).rename(columns={key: 'text'})

@op("Create prompt")
def create_prompt(input, *, save_as='prompt', template: ops.LongStr):
  assert template, 'Please specify the template. Refer to columns using the Jinja2 syntax.'
  t = jinja.from_string(template)
  prompt = t.render(**input)
  return {**input, save_as: prompt}

@op("Ask LLM")
def ask_llm(input, *, model: str, accepted_regex: str = None, max_tokens: int = 100):
  assert model, 'Please specify the model.'
  assert 'prompt' in input, 'Please create the prompt first.'
  options = {}
  if accepted_regex:
    options['extra_body'] = {
      "guided_regex": accepted_regex,
    }
  results = chat(
    model=model,
    max_tokens=max_tokens,
    messages=[
      {"role": "user", "content": input['prompt']},
    ],
    **options,
  )
  return [{**input, 'response': r} for r in results]

@op("View", view="table_view")
def view(input, *, _ctx: Context):
  v = _ctx.last_result
  if v:
    columns = v['dataframes']['df']['columns']
    v['dataframes']['df']['data'].append([input[c] for c in columns])
  else:
    columns = [str(c) for c in input.keys() if not str(c).startswith('_')]
    v = {
      'dataframes': { 'df': {
        'columns': columns,
        'data': [[input[c] for c in columns]],
      }}
    }
  return v

@ops.input_position(input="right")
@ops.output_position(output="left")
@op("Loop")
def loop(input, *, max_iterations: int = 3, _ctx: Context):
  '''Data can flow back here max_iterations-1 times.'''
  key = f'iterations-{_ctx.node.id}'
  input[key] = input.get(key, 0) + 1
  if input[key] < max_iterations:
    return input

@op('Branch', outputs=['true', 'false'])
def branch(input, *, expression: str):
  res = eval(expression, input)
  return Output(output_handle=str(bool(res)).lower(), value=input)

@ops.input_position(db="top")
@op('RAG')
def rag(input, db, *, input_field='text', db_field='text', num_matches: int=10, _ctx: Context):
  last = _ctx.last_result
  if last:
    collection = last['_collection']
  else:
    collection_name = _ctx.node.id.replace(' ', '_')
    for c in chroma_client.list_collections():
      if c.name == collection_name:
        chroma_client.delete_collection(name=collection_name)
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(
      documents=[r[db_field] for r in db],
      ids=[str(i) for i in range(len(db))],
    )
  results = collection.query(
    query_texts=[input[input_field]],
    n_results=num_matches,
  )
  results = [db[int(r)] for r in results['ids'][0]]
  return {**input, 'rag': results, '_collection': collection}

@op('Run Python')
def run_python(input, *, template: str):
  assert template, 'Please specify the template. Refer to columns using their names in uppercase.'
  p = template
  for k, v in input.items():
    p = p.replace(k.upper(), str(v))
  return p

EXECUTOR_OUTPUT_CACHE = {}

@ops.register_executor(ENV)
def execute(ws):
  catalog = ops.CATALOGS[ENV]
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
  for stage in get_stages(ws):
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
          key = json.dumps(fastapi.encoders.jsonable_encoder((inputs, params)))
          if key not in EXECUTOR_OUTPUT_CACHE:
            EXECUTOR_OUTPUT_CACHE[key] = op.func(*inputs, **params)
          result = EXECUTOR_OUTPUT_CACHE[key]
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

def df_to_list(df):
  return [dict(zip(df.columns, row)) for row in df.values]

def has_ctx(op):
  sig = inspect.signature(op.func)
  return '_ctx' in sig.parameters

def get_stages(ws):
  '''Inputs on top are batch inputs. We decompose the graph into a DAG of components along these edges.'''
  catalog = ops.CATALOGS[ENV]
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
