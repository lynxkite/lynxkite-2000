---
title: LynxKite 2000:MM
emoji: 🪁
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

# LynxKite 2024

This is an experimental rewrite of [LynxKite](https://github.com/lynxkite/lynxkite). It is not compatible with the
original LynxKite. The primary goals of this rewrite are:

- Target GPU clusters instead of Hadoop clusters. We use Python instead of Scala, RAPIDS instead of Apache Spark.
- More extensible backend. Make it easy to add new LynxKite boxes. Make it easy to use our frontend for other purposes,
  configuring and executing other pipelines.

## Structure

- `lynxkite-core`: Core types and utilities. Depend on this lightweight package if you are writing LynxKite plugins.
- `lynxkite-app`: The LynxKite web application. Install some plugins then run this to use LynxKite.
- `lynxkite-graph-analytics`: Graph analytics plugin. The classical LynxKite experience!
- `lynxkite-pillow`: A simple example plugin.
- `docs`: User-facing documentation. It's shared between all packages.

## Development

Install everything like this:

```bash
uv venv
source .venv/bin/activate
uvx pre-commit install
# The [dev] tag is only needed if you intend on running tests
uv pip install -e lynxkite-core/[dev] -e lynxkite-app/[dev] -e lynxkite-graph-analytics/[dev] -e lynxkite-pillow-example/
```

This also builds the frontend, hopefully very quickly. To run it:

```bash
cd examples && lynxkite
```

If you also want to make changes to the frontend with hot reloading:

```bash
cd lynxkite-app/web
npm run dev
```

## Executing tests

Run all tests with a single command, or look inside to see how to run them individually:

```bash
./test.sh
```

## Documentation

To work on the documentation:

```bash
uv pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```
