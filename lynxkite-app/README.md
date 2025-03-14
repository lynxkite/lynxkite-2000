# LynxKite MM

This is an experimental rewrite of [LynxKite](https://github.com/lynxkite/lynxkite). It is not compatible with the
original LynxKite. The primary goals of this rewrite are:

- Target GPU clusters instead of Hadoop clusters. We use Python instead of Scala, RAPIDS instead of Apache Spark.
- More extensible backend. Make it easy to add new LynxKite boxes. Make it easy to use our frontend for other purposes,
  configuring and executing other pipelines.

## Development

To run the backend:

```bash
uv pip install -e .
cd ../examples && LYNXKITE_RELOAD=1 lynxkite
```

To run the frontend:

```bash
cd web
npm i
npm run dev
```

To update the frontend types with the backend types:

```bash
$ uv run pydantic2ts --module lynxkite_app.main --output ./web/src/apiTypes.ts --json2ts-cmd "npx json-schema-to-typescript"
```
