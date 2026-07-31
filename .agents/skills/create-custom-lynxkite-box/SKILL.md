---
name: create-custom-lynxkite-box
description: Create a new type of box with custom behavior in boxes.py to create opearations not yet defined in LynxKite.
---

## Adding new operations

Any piece of Python code can easily be wrapped into a LynxKite operation.
Let's say we have some code that calculates the length of a string column in a Pandas DataFrame:

```python
df["length"] = df["my_column"].str.len()
```

We can turn it into a LynxKite operation using the
`@op` decorator:

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

The `@op` decorator registers a function as a LynxKite operation. You must specify the name of the operation as an argument.

The `op_registration` function is already included in `boxes.py` with the correct environment setting: do not change it.

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
- `lynxkite_graph_analytics.Bundle`: see ./references/types.md for more information

The inputs of an operation are automatically converted to the right type, when possible.

To make an input optional, use an optional type, like `pd.DataFrame | None`.

The position of the input and output connectors can be controlled using the
`@ops.input_position` and `@ops.output_position` decorators. By default, inputs are on the left and outputs on the right.

All **parameters** are stored in LynxKite workspaces as strings. If a type annotation is provided,
LynxKite will convert the string to the right type and provide the right UI.

- `str`, `int`, `float` are presented as a text box and converted to the given type.
- `bool` is presented as a checkbox.
- `lynxkite_core.ops.LongStr`] is presented as a text area.
- Enums are presented as a dropdown list.
- Pydantic models are presented as their JSON string representations. (Unless you add custom UI for them.) They are converted to the model object when your function is called.
- see ./references/types.md for more parameter types, such as dropdowns menus

### Slow operations

If the function takes a significant amount of time to run, we must either:

- Write an asynchronous function.
- Pass `slow=True` to the `@op` decorator. LynxKite will run the function in a separate thread.

`slow=True` also causes the results of the operation to be cached on disk. As long as
its inputs don't change, the operation will not be run again. This is useful for both
synchronous and synchronous operations.

### Documentation

The docstring of the function is used as the operation's description. You can use Google-style or Numpy-style docstrings.
(See [Griffe's documentation](https://mkdocstrings.github.io/griffe/reference/docstrings/).)

The docstring should be omitted for simple operations like the one above.

### Outputting results

The return value of the function is the output of the operation. It will be passed to the
next operation in the pipeline.

An operation can have multiple outputs. In this case, the return value must be a dictionary,
and the list of outputs must be declared in the `@op` decorator.

```python
op = ops.op_registration("LynxKite Graph Analytics")

@op("Train/test split", outputs=["train", "test"])
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
    op = ops.op_registration("LynxKite Graph Analytics")

    @op("Plot column histogram", view="matplotlib")
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
- `table_view`: Return `BundleTableView()`: see ./references/types.md for more information
