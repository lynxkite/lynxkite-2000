# LynxScribe in LynxKite

LynxKite UI for building LynxScribe chat applications. Also runs the chat application!

To run a chat UI for LynxScribe workspaces:

```bash
WEBUI_AUTH=false OPENAI_API_BASE_URL=http://localhost:8000/api/service/server.lynxscribe_ops uvx open-webui serve
```

Or use [Lynx WebUI](https://github.com/biggraph/lynx-webui/) instead of Open WebUI.

Run tests with:

```bash
uv run pytest
```
