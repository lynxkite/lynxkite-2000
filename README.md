# LynxKite 2000:MM

LynxKite 2000:MM is a GPU-accelerated data science platform and a general tool for collaboratively edited workflows.

Features include:

- A web UI for building and executing data science workflows.
- An extensive toolbox of graph analytics operations powered by NVIDIA RAPIDS (CUDA).
- An integrated collaborative code editor makes it easy to add new operations.
- An environment for visually designing neural network model architectures.
- The infrastructure for easily creating other workflow design environments. See `lynxkite-pillow-example` for a simple example.

This is the next evolution of the classical [LynxKite](https://github.com/lynxkite/lynxkite).
The two tools offer similar functionality, but are not compatible.
This version runs on GPU clusters instead of Hadoop clusters.
It targets CUDA instead of Apache Spark. It is much more extensible.

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

## License

LynxKite 2000:MM is licensed under the GNU AGPLv3. See the [LICENSE](LICENSE) file for details.

[Lynx Analytics](https://www.lynxanalytics.com/) offers a commercial license of LynxKite 2000:MM
that includes additional features and support. Get in touch if you are interested in life sciences tools
and cluster deployment!
