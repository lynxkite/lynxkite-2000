'''For specifying an LLM agent logic flow.'''
from . import ops
import json
import openai
import pandas as pd

client = openai.OpenAI(base_url="http://localhost:11434/v1")
CACHE = {}

def chat(*args, **kwargs):
  key = json.dumps({'args': args, 'kwargs': kwargs})
  if key not in CACHE:
    completion = client.chat.completions.create(*args, **kwargs)
    CACHE[key] = [c.message.content for c in completion.choices]
  return CACHE[key]

@ops.op("Input")
def input(*, filename: ops.PathStr, key: str):
  return pd.read_csv(filename).rename(columns={key: 'text'})

@ops.op("Create prompt")
def create_prompt(input, *, template: ops.LongStr):
  assert template, 'Please specify the template. Refer to columns using their names in uppercase.'
  df = input.copy()
  prompts = []
  for i, row in df.iterrows():
    p = template
    for c in df.columns:
      p = p.replace(c.upper(), str(row[c]))
    prompts.append(p)
  df['prompt'] = prompts
  return df


@ops.op("Ask LLM")
def ask_llm(input, *, model: str, choices: list = None, max_tokens: int = 100):
  assert model, 'Please specify the model.'
  assert 'prompt' in input.columns, 'Please create the prompt first.'
  df = input.copy()
  g = {}
  if choices:
    g['extra_body'] = {
      "guided_choice": choices.split()
    }
  for i, row in df.iterrows():
    [res] = chat(
      model=model,
      max_tokens=max_tokens,
      messages=[
        {"role": "user", "content": row['prompt']},
      ],
      **g,
    )
    df.loc[i, 'response'] = res
  return df

@ops.op("View", view="table_view")
def view(input):
  v = {
    'dataframes': { 'df': {
      'columns': [str(c) for c in input.columns],
      'data': input.values.tolist(),
    }}
  }
  return v

@ops.input_position(input="right")
@ops.output_position(output="left")
@ops.op("Loop")
def loop(input, *, max_iterations: int = 10):
  '''Data can flow back here until it becomes empty or reaches the limit.'''
  return input
