'''API for implementing LynxKite operations.'''
import dataclasses
import inspect

ALL_OPS = {}

@dataclasses.dataclass
class Op:
  func: callable
  name: str
  params: dict
  inputs: dict
  outputs: dict
  type: str

  def __call__(self, *inputs, **params):
    # Convert parameters.
    sig = inspect.signature(self.func)
    for p in params:
      if p in self.params:
        t = sig.parameters[p].annotation
        if t == int:
          params[p] = int(params[p])
    res = self.func(*inputs, **params)
    return res

@dataclasses.dataclass
class EdgeDefinition:
  df: str
  source_column: str
  target_column: str
  source_table: str
  target_table: str
  source_key: str
  target_key: str

@dataclasses.dataclass
class Bundle:
  dfs: dict
  edges: list[EdgeDefinition]

def op(name):
  '''Decorator for defining an operation.'''
  def decorator(func):
    type = func.__annotations__.get('return') or 'basic'
    sig = inspect.signature(func)
    # Positional arguments are inputs.
    inputs = {
      name: param.annotation
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {
      name: param.default if param.default is not inspect._empty else None
      for name, param in sig.parameters.items()
      if param.kind == param.KEYWORD_ONLY}
    op = Op(func, name, params=params, inputs=inputs, outputs={'output': 'yes'}, type=type)
    ALL_OPS[name] = op
    return func
  return decorator
