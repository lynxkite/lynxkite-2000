'''For specifying an LLM agent logic flow.'''
from . import ops
import chromadb
import jinja2
import json
import openai
import pandas as pd
from .executors import one_by_one

client = openai.OpenAI(base_url="http://localhost:11434/v1")
jinja = jinja2.Environment()
chroma_client = chromadb.Client()
LLM_CACHE = {}
ENV = 'LLM logic'
one_by_one.register(ENV)
op = ops.op_registration(ENV)

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
def view(input, *, _ctx: one_by_one.Context):
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
def loop(input, *, max_iterations: int = 3, _ctx: one_by_one.Context):
  '''Data can flow back here max_iterations-1 times.'''
  key = f'iterations-{_ctx.node.id}'
  input[key] = input.get(key, 0) + 1
  if input[key] < max_iterations:
    return input

@op('Branch', outputs=['true', 'false'])
def branch(input, *, expression: str):
  res = eval(expression, input)
  return one_by_one.Output(output_handle=str(bool(res)).lower(), value=input)

@ops.input_position(db="top")
@op('RAG')
def rag(input, db, *, input_field='text', db_field='text', num_matches: int=10, _ctx: one_by_one.Context):
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


