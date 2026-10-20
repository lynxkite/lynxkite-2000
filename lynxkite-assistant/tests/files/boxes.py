"""Custom box definitions for the workspace.

To add a custom box, define a function here and decorate it with @op.
The positional arguments of the function become its inputs, and the keyword-only arguments become its parameters.
E.g.:

    @op("Blur")
    def blur(image: Image.Image, *, radius = 5):
        return image.filter(ImageFilter.GaussianBlur(radius))

    @op("Read CSV")
    def read_csv(*, path: str):
        return pd.read_csv(path)

To use them in the workspace, call them in `workspace.py` with this custom module name: boxes.
For example:
    boxes.blur(...)
    boxes.read_csv(...)
"""

from lynxkite_core import ops
import lynxkite_graph_analytics
import numpy as np

op = ops.op_registration("LynxKite Graph Analytics")


@op("Copy column from selector")
def copy_column_from_selector(b, *, tc: lynxkite_graph_analytics.core.TableColumn, n="sel"):
    t, c = tc
    bc = b.dfs[t].copy()
    col_indices = bc.columns.get_indexer(bc[c])
    row_indices = np.arange(len(bc))
    result = bc.values[row_indices, col_indices]
    bc["sel"] = result
    return bc


# Add new box definitions here.
