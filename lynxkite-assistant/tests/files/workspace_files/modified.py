"""The Python representation of the workspace."""

"""All imports are handled automatically. Do not add imports."""
res_enter_table_data_1 = lynxkite_graph_analytics.operations.table_ops.enter_table_data(
    data="c,x,y,z\nx,1,2,3\nx,0,6,7\nz,2,3,8", table_name="table"
)
#! ## A beautiful comment
#!
#! Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

lynxkite_graph_analytics.operations.visualization_ops.scatter_plot(
    b=res_enter_table_data_1, x=["table", "x"], y=["table", "y"]
)


selected_res = lynxkite_assistant.tests.files.boxes.copy_column_from_selector(
    b=res_enter_table_data_1, tc=("table", "c")
)
lynxkite_graph_analytics.operations.basic_ops.view_tables(bundle=selected_res)
