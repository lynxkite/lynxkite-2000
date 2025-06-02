# Contributing

The LynxKite 2000:MM repository lives at
[https://github.com/lynxkite/lynxkite-2000](https://github.com/lynxkite/lynxkite-2000). Bug reports, feature requests,
and pull requests are welcome!

## Project structure

- `lynxkite-core`: Core types and utilities. Depend on this lightweight package if you are writing LynxKite plugins.
- `lynxkite-app`: The LynxKite web application. Install some plugins then run this to use LynxKite.
- `lynxkite-graph-analytics`: Graph analytics plugin. The classical LynxKite experience!
- `lynxkite-pillow`: A simple example plugin.
- `lynxkite-lynxscribe`: A plugin for building and running LynxScribe applications.
- `lynxkite-bio`: Bioinformatics additions for LynxKite Graph Analytics.
- `docs`: User-facing documentation. It's shared between all packages.

## Development setup

Install everything like this:

```bash
uv venv
source .venv/bin/activate
uvx pre-commit install
uv pip install -e 'lynxkite-core/[dev]' -e 'lynxkite-app/[dev]' -e 'lynxkite-graph-analytics/[dev]' -e lynxkite-pillow-example/ -e lynxkite-bio -e lynxkite-lynxscribe/
```

This also builds the frontend, hopefully very quickly. To run it:

```bash
cd examples
lynxkite
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
