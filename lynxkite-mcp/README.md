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


# Assistant MCP

An MCP server that provides tools for an AI agent enabling the agent to edit workspaces.
The agent will have all the capabilities of the LynxKite Assistant, and it will also be
able to create new workspaces.

For now, this has only been tested in a VSCode environment, where the built-in agent can
automatically connect to MCP servers defined in .vscode/mcp.json.

## Setup in VSCode

You can either run VSCode in the parent or a sibling directory of the lynxkite-2000 repo.
In the first case, the agent will be able to see the workspace files directly, and will
have access to more information.

### Setup in the parent directory:

You can simply open the parent directory of the repo in VSCode (or an empty directory,
if you do not have the repo cloned yet) and copy-paste the following as a prompt:
>If the repo lynxkite-2000 is not yet cloned in this directory, run `git clone https://github.com/lynxkite/lynxkite-2000.git`. Afterwards, run the agent environment setup `./lynxkite-2000/lynxkite-mcp/agent_env_setup/setup` bash script. You should be able to see a new .agents folder with skills, instructions in AGENTS.md and an MCP server defined in .vscode/mcp.json. If you're not running in VSCode, make sure you start the MCP server (in VSCode, this should start automatically).
>Ask the user if they want to lauch the LynxKite application as well in order to see the workspaces. If yes, launch the app by running
>- If .venv already exists in lynxkite-2000
>    ```
>    cd lynxkite-2000/examples
>    source ../.venv/bin/activate
>    lynxkite
>    ```
>- If the virtual environment doesn't exist yet, run:
>    ```
>    cd lynxkite-2000
>    uv venv
>    source .venv/bin/activate
>    uvx pre-commit install
>    uv sync
>    cd examples
>    lynxkite
>    ```

### Setup in sibling directory
You can also choose to set the agent's working directory as a sibling of the
lynxkite-2000 repo, to ensure that the agent only has access to the workspace
files throught the MCP tools.

To do this, run this script from the repo's parent directory:
`./lynxkite-2000/lynxkite-mcp/agent_env_setup/setup <name_of_sibling_directory>`.
This will create the sibling directory and the environment for the agent. Open the
sibling directory in VSCode and the MCP server should start automatically.
