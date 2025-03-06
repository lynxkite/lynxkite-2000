# LynxScribe in LynxKite

LynxKite UI for building LynxScribe chat applications. Also runs the chat application!

To run a chat UI for LynxScribe workspaces:

```bash
WEBUI_AUTH=false OPENAI_API_BASE_URL=http://localhost:8000/api/service/lynxkite_lynxscribe uvx open-webui serve
```

Or use [Lynx WebUI](https://github.com/biggraph/lynx-webui/) instead of Open WebUI.

Run tests with:

```bash
uv run pytest
```

The LLM agent flow examples use local models.

```bash
uv pip install infinity-emb[all]
infinity_emb v2 --model-id michaelfeil/bge-small-en-v1.5
uv pip install "sglang[all]>=0.4.2.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
python -m sglang.launch_server --model-path SultanR/SmolTulu-1.7b-Instruct --port 8080
```
