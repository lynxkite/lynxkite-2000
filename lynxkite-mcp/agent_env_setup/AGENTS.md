# Working with LynxKite workspaces

- LynxKite workspaces use the .lynxkite extension; before startup, migrate legacy .lynxkite.json workspaces with `python3 scripts/migrate_workspaces.py <DATA_DIR>`
- reading and writing these workspaces MUST BE done via LynxKite MCP tools
- DO NOT edit .lynxkite workspaces directly !!!
- when using MCP tools, always use the relative path the user specified, not an absolute path
