# Working with LynxKite workspaces

- LynxKite workspaces use the .lynxkite extension; it's a directory containing the workspace.json (which houses the workspace) and an additional node_data directory which has every node's data (e.g. display.json).
- reading and writing these workspaces MUST BE done via LynxKite MCP tools
- DO NOT edit .lynxkite workspaces directly !!!
- when using MCP tools, always use the relative path the user specified, not an absolute path
