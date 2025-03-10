FROM node:22
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y git
# ADD pyproject.toml /repo/pyproject.toml
# ADD lynxkite-app/pyproject.toml /repo/lynxkite-app/pyproject.toml
# ADD lynxkite-core/pyproject.toml /repo/lynxkite-core/pyproject.toml
# ADD lynxkite-graph-analytics/pyproject.toml /repo/lynxkite-graph-analytics/pyproject.toml
# ADD lynxkite-bio/pyproject.toml /repo/lynxkite-bio/pyproject.toml
# ADD lynxkite-pillow-example/pyproject.toml /repo/lynxkite-pillow-example/pyproject.toml
ADD . /repo
WORKDIR /repo
RUN uv venv && uv pip install \
  -e lynxkite-core \
  -e lynxkite-app \
  -e lynxkite-graph-analytics \
  -e lynxkite-bio \
  -e lynxkite-pillow-example
ENV LYNXKITE_DATA=examples
ENV PORT=7860
CMD ["uv", "run", "--no-sync", "lynxkite"]
