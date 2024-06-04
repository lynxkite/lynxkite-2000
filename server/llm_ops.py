'''For specifying an LLM agent logic flow.'''
from . import ops
import dataclasses
import inspect
import json
import openai
import pandas as pd
import traceback
from . import workspace

client = openai.OpenAI(base_url="http://localhost:11434/v1")
CACHE = {}
ENV = 'LLM logic'
op = ops.op_registration(ENV)

@dataclasses.dataclass
class Context:
  '''Passed to operation functions as "_ctx" if they have such a parameter.'''
  node: workspace.WorkspaceNode
  last_result = None

@dataclasses.dataclass
class Output:
  '''Return this to send values to specific outputs of a node.'''
  output_handle: str
  value: dict

def chat(*args, **kwargs):
  key = json.dumps({'args': args, 'kwargs': kwargs})
  if key not in CACHE:
    completion = client.chat.completions.create(*args, **kwargs)
    CACHE[key] = [c.message.content for c in completion.choices]
  return CACHE[key]

@op("Input")
def input(*, filename: ops.PathStr, key: str):
  return pd.read_csv(filename).rename(columns={key: 'text'})

@op("Create prompt")
def create_prompt(input, *, template: ops.LongStr):
  assert template, 'Please specify the template. Refer to columns using their names in uppercase.'
  p = template
  for k, v in input.items():
    p = p.replace(k.upper(), str(v))
  return p

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
        'data': [input[c] for c in columns],
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
  return Output(str(bool(res)).lower(), input)

@ops.input_position(db="top")
@op('RAG')
def rag(input, db, *, closest_n: int=10):
  return input

@op('Run Python')
def run_python(input, *, template: str):
  assert template, 'Please specify the template. Refer to columns using their names in uppercase.'
  p = template
  for k, v in input.items():
    p = p.replace(k.upper(), str(v))
  return p



@ops.register_executor(ENV)
def execute(ws):
  catalog = ops.CATALOGS[ENV]
  nodes = {n.id: n for n in ws.nodes}
  contexts = {n.id: Context(n) for n in ws.nodes}
  edges = {n.id: [] for n in ws.nodes}
  for e in ws.edges:
    edges[e.source].append(e.target)
  tasks = {}
  NO_INPUT = object() # Marker for initial tasks.
  for node in ws.nodes:
    node.data.error = None
    op = catalog[node.data.title]
    # Start tasks for nodes that have no inputs.
    if not op.inputs:
      tasks[node.id] = [NO_INPUT]
  # Run the rest until we run out of tasks.
  while tasks:
    n, ts = tasks.popitem()
    node = nodes[n]
    data = node.data
    op = catalog[data.title]
    params = {**data.params}
    if has_ctx(op):
      params['_ctx'] = contexts[node.id]
    results = []
    for task in ts:
      try:
        if task is NO_INPUT:
          result = op(**params)
        else:
          # TODO: Tasks with multiple inputs?
          result = op(task, **params)
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
        data.display = results
      for target in edges[node.id]:
        tasks.setdefault(target, []).extend(results)

def df_to_list(df):
  return [dict(zip(df.columns, row)) for row in df.values]

def has_ctx(op):
  sig = inspect.signature(op.func)
  return '_ctx' in sig.parameters
