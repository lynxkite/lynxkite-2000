# Plugin development

Plugins can provide additional operations for an existing LynxKite environment,
and they can also provide new environments.

## Creating a new plugin

`.py` files inside the LynxKite data directory are automatically imported each time a
workspace is executed. You can create a new plugin by creating a new `.py` file in the
data directory. LynxKite even includes an integrated editor for this purpose.
Click **New code file** in the directory where you want to create the file.

Plugins in subdirectories of the data directory are imported when executing workspaces
within those directories. This allows you to create plugins that are only available
in specific workspaces.

You can also create and distribute plugins as Python packages. In this case the
module name must start with `lynxkite_` for it to be automatically imported on startup.

### Plugin dependencies

When creating a plugin as a "code file", you can create a `requirements.txt` file in the same
directory. This file will be used to install the dependencies of the plugin.

## Adding new operations

Any piece of Python code can easily be wrapped into a LynxKite operation.
Let's say we have some code that calculates the length of a string column in a Pandas DataFrame:

```python
df["length"] = df["my_column"].str.len()
```

We can turn it into a LynxKite operation using the
[`@op`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.op) decorator:

```python
import pandas as pd
from lynxkite_core.ops import op

@op("LynxKite Graph Analytics", "Get column length")
def get_length(df: pd.DataFrame, *, column_name: str):
    """
    Gets the length of a string column.

    Args:
        column_name: The name of the column to get the length of.
    """
    df = df.copy()
    df["length"] = df[column_name].str.len()
    return df
```

Let's review the changes we made.

### The `@op` decorator

The [`@op`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.op) decorator registers a
function as a LynxKite operation. The first argument is the name of the environment,
the last argument is the name of the operation. Between the two, you can list the hierarchy of
categories the operation belongs to. For example:

```python
@op("LynxKite Graph Analytics", "Machine learning", "Preprocessing", "Split train/test set")
```

When defining multiple operations, you can use
[`ops.op_registration`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.op_registration)
for convenience:
```python
op = ops.op_registration("LynxKite Graph Analytics")

@op("An operation")
def my_op():
    ...
```

### The function signature

`*` in the list of function arguments marks the start of keyword-only arguments.
The arguments before `*` will become _inputs_ of the operation. The arguments after `*` will
be its _parameters_.

```python
#              /--- inputs ---\     /- parameters -\
def get_length(df: pd.DataFrame, *, column_name: str):
```

LynxKite uses the type annotations of the function arguments to provide input validation,
conversion, and the right UI on the frontend.

The types supported for **inputs** are determined by the environment. For graph analytics,
the possibilities are:

- `pandas.DataFrame`
- `networkx.Graph`
- [`lynxkite_graph_analytics.Bundle`](../reference/lynxkite-graph-analytics/core.md#lynxkite_graph_analytics.core.Bundle)

The inputs of an operation are automatically converted to the right type, when possible.

To make an input optional, use an optional type, like `pd.DataFrame | None`.

The position of the input and output connectors can be controlled using the
[`@ops.input_position`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.input_position) and
[`@ops.output_position`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.output_position)
decorators. By default, inputs are on the left and outputs on the right.

All **parameters** are stored in LynxKite workspaces as strings. If a type annotation is provided,
LynxKite will convert the string to the right type and provide the right UI.

- `str`, `int`, `float` are presented as a text box and converted to the given type.
- `bool` is presented as a checkbox.
- [`lynxkite_core.ops.LongStr`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.LongStr)
  is presented as a text area.
- Enums are presented as a dropdown list.
- Pydantic models are presented as their JSON string representations. (Unless you add custom UI
  for them.) They are converted to the model object when your function is called.

### Slow operations

If the function takes a significant amount of time to run, we must either:

- Write an asynchronous function.
- Pass `slow=True` to the `@op` decorator. LynxKite will run the function in a separate thread.

`slow=True` also causes the results of the operation to be cached on disk. As long as
its inputs don't change, the operation will not be run again. This is useful for both
synchronous and synchronous operations.

### Documentation

The docstring of the function is used as the operation's description. You can use
Google-style or Numpy-style docstrings.
(See [Griffe's documentation](https://mkdocstrings.github.io/griffe/reference/docstrings/).)

The docstring should be omitted for simple operations like the one above.

### Outputting results

The return value of the function is the output of the operation. It will be passed to the
next operation in the pipeline.

An operation can have multiple outputs. In this case, the return value must be a dictionary,
and the list of outputs must be declared in the `@op` decorator.

```python
@op("LynxKite Graph Analytics", "Train/test split", outputs=["train", "test"])
def test_split(df: pd.DataFrame, *, test_ratio=0.1):
    test = df.sample(frac=test_ratio).reset_index()
    train = df.drop(test.index).reset_index()
    return {"train": train, "test": test}
```

### Displaying results

The outputs of the operation can be used by other operations. But we can also generate results
that are meant to be viewed by the user. The different options for this are controlled by the `view`
argument of the `@op` decorator.

The `view` argument can be one of the following:

- `matplotlib`: Just plot something with Matplotlib and it will be displayed in the UI.

    ```python
    @op("LynxKite Graph Analytics", "Plot column histogram", view="matplotlib")
    def plot(df: pd.DataFrame, *, column_name: str):
        df[column_name].value_counts().sort_index().plot.bar()
    ```

- `visualization`: Draws a chart using [ECharts](https://echarts.apache.org/examples/en/index.html).
  You need to return a dictionary with the chart configuration, which ECharts calls `option`.

    ```python
    @op("View loss", view="visualization")
    def view_loss(bundle: core.Bundle):
        loss = bundle.dfs["training"].training_loss.tolist()
        v = {
            "title": {"text": "Training loss"},
            "xAxis": {"type": "category"},
            "yAxis": {"type": "value"},
            "series": [{"data": loss, "type": "line"}],
        }
        return v
    ```

- `image`: Return an image as a
  [data URL](https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Schemes/data)
  and it will be displayed.
- `molecule`: Return a molecule as a PDB or SDF string, or an `rdkit.Chem.Mol` object.
  It will be displayed using [3Dmol.js](https://3Dmol.org/).
- `table_view`: Return
  [`Bundle.to_dict()`](../reference/lynxkite-graph-analytics/core.md#lynxkite_graph_analytics.core.Bundle.to_dict).

## Adding new environments

A new environment means a completely new set of operations, and (optionally) a new
executor. There's nothing to be done for setting up a new environment. Just start
registering operations into it.

### No executor

By default, the new environment will have no executor. This can be useful!

LynxKite workspaces are stored as straightforward JSON files and updated on every modification.
You can use LynxKite for configuring workflows and have a separate system
read the JSON files.

Since the code of the operations is not executed in this case, you can create functions that do nothing.
Alternatively, you can use the
[`register_passive_op`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.register_passive_op)
and
[`passive_op_registration`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.passive_op_registration)
functions to easily whip up a set of operations:

```python
from lynxkite_core.ops import passive_op_registration, Parameter as P

op = passive_op_registration("My Environment")
op('Scrape documents', params=[P('url', '')])
op('Conversation logs')
op('Extract graph')
op('Compute embeddings', params=[P.options('method', ['LLM', 'graph', 'random']), P('dimensions', 1234)])
op('Vector DB', params=[P.options('backend', ['ANN', 'HNSW'])])
op('Chat UI', outputs=[])
op('Chat backend')
```

### Built-in executors

LynxKite comes with two built-in executors. You can register these for your environment
and you're good to go.

```python
from lynxkite_core.executors import simple
simple.register("My Environment")
```

The [`simple` executor](../reference/lynxkite-core/executors/simple.md)
runs each operation once, passing the output of the preceding operation
as the input to the next one. No tricks. You can use any types as inputs and outputs.

```python
from lynxkite_core.executors import one_by_one
one_by_one.register("My Environment")
```

The [`one_by_one` executor](../reference/lynxkite-core/executors/one_by_one.md)
expects that the code for operations is the code for transforming
a single element. If an operation returns an iterable, it will be split up
into its elements, and the next operation is called for each element.

Sometimes you need the full contents of an input. The `one_by_one` executor
lets you choose between the two modes by the orientation of the input connector.
If the input connector is horizontal (left or right), it takes single elements.
If the input connector is vertical (top or bottom), it takes an iterable of all the incoming data.

A unique advantage of this setup is that horizontal inputs can have loops across
horizontal inputs. Just make sure that loops eventually discard all elements, so you don't
end up with an infinite loop.

### Custom executors

A custom executor can be registered using
[`@ops.register_executor`](../reference/lynxkite-core/ops.md#lynxkite_core.ops.register_executor).

```python
@ops.register_executor(ENV)
async def execute(ws: workspace.Workspace, ctx: workspace.WorkspaceExecutionContext | None):
    catalog = ops.CATALOGS[ws.env]
    ...
```

The executor must be an asynchronous function that takes a
[`workspace.Workspace`](../reference/lynxkite-core/workspace.md#lynxkite_core.workspace.Workspace)
as an argument. The return value is ignored and it's up to you how you process the workspace.

To update the frontend as the executor processes the workspace, call
[`WorkspaceNode.publish_started`](../reference/lynxkite-core/workspace.md#lynxkite_core.workspace.WorkspaceNode.publish_started)
when starting to execute a node, and
[`WorkspaceNode.publish_result`](../reference/lynxkite-core/workspace.md#lynxkite_core.workspace.WorkspaceNode.publish_result)
to publish the results. Use
[`WorkspaceNode.publish_error`](../reference/lynxkite-core/workspace.md#lynxkite_core.workspace.WorkspaceNode.publish_error)
if the node failed.
