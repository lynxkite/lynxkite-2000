"""Basic operations for this environment."""

from lynxkite_core import ops
from .. import core

op = ops.op_registration(core.ENV)


@op("View tables", view="table_view", color="blue", icon="table-filled")
def view_tables(bundle: core.Bundle, *, _tables_open: str = "", limit: int = 100):
    _tables_open = _tables_open  # The frontend uses this parameter to track which tables are open.
    return bundle.to_table_view(limit=limit)
