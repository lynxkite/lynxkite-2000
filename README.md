# LynxKite 2024

This is an experimental rewrite of [LynxKite](https://github.com/lynxkite/lynxkite). It is not compatible with the
original LynxKite. The primary goals of this rewrite are:

- Target GPU clusters instead of Hadoop clusters. We use Python instead of Scala, RAPIDS instead of Apache Spark.
- More extensible backend. Make it easy to add new LynxKite boxes. Make it easy to use our frontend for other purposes,
  configuring and executing other pipelines.

## Installation

To run the backend:

```bash
PYTHONPATH=. uv run pydantic2ts --module server.workspace --output ./web/src/apiTypes.ts --json2ts-cmd "npm exec --prefix web json2ts"
uv run fastapi run server/main.py --reload
```

To run the frontend:

```bash
cd web
npm i
npm run dev
```

To run a chat UI for LynxScribe workspaces:

```bash
WEBUI_AUTH=false OPENAI_API_BASE_URL=http://localhost:8000/api/service/server.lynxscribe_ops uvx open-webui serve
```
