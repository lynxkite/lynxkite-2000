# LynxKite MCP

An MCP server for running a LynxKite workspace. The boxes with no inputs, such as
"Import CSV" are considered the inputs of the workspace, and the boxes with no outputs,
such as "View table" are considered its outputs. The model using the MCP server
will be able to modify the settings in the input boxes and access the results in the
output boxes.

In the MCP client, configure a command such as:

```
lynxkite-mcp 'examples/NetworkX demo.lynxkite.json'
```

This will take care of running the workspace. No need for a running LynxKite instance.
