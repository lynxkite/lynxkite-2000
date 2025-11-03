"""Expose a single LynxKite workspace as an MCP tool."""

import sys
from lynxkite_core import ops, workspace
from mcp.server.fastmcp import FastMCP, tools
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, ArgModelBase
import pydantic
import pydantic.fields


class WorkspaceAsTool:
    def __init__(self, ws_path: str):
        self.ws_path = ws_path
        self.load_workspace()

    def load_workspace(self):
        self.ws = workspace.Workspace.load(self.ws_path)
        edges_in = {}
        for e in self.ws.edges:
            edges_in.setdefault(e.target, []).append(e.source)
        edges_out = {}
        for e in self.ws.edges:
            edges_out.setdefault(e.source, []).append(e.target)
        input_nodes = []
        self.output_nodes = set()
        func_name = self.ws_path.rsplit("/", 1)[-1].removesuffix(".lynxkite.json")
        description = ""
        for n in self.ws.nodes:
            if n.data.op_id == "Comment":
                comment = n.data.params["text"].strip()
                if len(comment) > len(description):
                    description = comment
                continue
            if n.data.collapsed:
                continue
            if n.id not in edges_in:
                input_nodes.append(n)
            if n.id not in edges_out:
                self.output_nodes.add(n.id)
        model_fields = {}
        self.exposed_params = {}
        for n in input_nodes:
            op: ops.Op | None = ops.CATALOGS[self.ws.env].get(n.data.op_id)
            if not op:
                continue
            docs = {}
            for d in op.doc or []:
                if d["kind"] == "parameters":
                    for p in d["value"]:
                        docs[p["name"]] = p["description"]
            converted_params = op.convert_params(n.data.params)
            for p in op.params:
                if isinstance(p, ops.ParameterGroup):
                    # Not yet supported.
                    continue
                field_info = pydantic.fields.FieldInfo.from_annotation(p.type)
                field_info.default = converted_params.get(p.name, p.default)
                if p.name in docs:
                    field_info.description = docs[p.name]
                model_fields[p.name] = field_info.annotation, field_info
                self.exposed_params[p.name] = n.id
        arg_model = pydantic.create_model("Arguments", **model_fields, __base__=ArgModelBase)
        self.mcp_tool = tools.Tool(
            fn=self.run,
            is_async=True,
            name=func_name,
            description=description,
            parameters=arg_model.model_json_schema(),
            fn_metadata=FuncMetadata(arg_model=arg_model),
        )

    def with_params(self, params: dict):
        ws = self.ws.model_copy(deep=True)
        nodes = {n.id: n for n in ws.nodes}
        for k, v in params.items():
            if k not in self.exposed_params:
                continue
            n_id = self.exposed_params[k]
            if n_id not in nodes:
                continue
            n = nodes[n_id]
            n.data.params[k] = str(v)
        return ws

    async def run(self, **params):
        self.load_workspace()
        ws = self.with_params(params)
        await ws.execute()
        return get_node_outputs([n for n in ws.nodes if n.id in self.output_nodes])


def get_node_outputs(nodes: list[workspace.WorkspaceNode]):
    result = {}
    for n in nodes:
        data = simplify_output(n)
        if data:
            result[n.id] = data
    if len(result) == 1:
        [result] = result.values()
    return result


def simplify_output(n: workspace.WorkspaceNode):
    display = n.data.display
    if not display:
        return None
    if n.data.meta and n.data.meta.type == "table_view":
        tables_open = n.data.params.get("_tables_open", [])
        if len(tables_open) == 1:
            [table_name] = tables_open
            table = display["dataframes"][table_name]
            if len(table["data"]) == 1:
                [record] = table["data"]
                return {c: v for c, v in zip(table["columns"], record)}
            return table
    return display


def main():
    ops.detect_plugins()
    ws_path = sys.argv[1]
    tool = WorkspaceAsTool(ws_path)
    mcp = FastMCP(ws_path, tools=[tool.mcp_tool])
    mcp.run(transport="stdio")
