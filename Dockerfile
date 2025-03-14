FROM node:22
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y git
USER node
ENV HOME=/home/node PATH=/home/node/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=node . $HOME/app
RUN uv venv && uv pip install \
  -e lynxkite-core \
  -e lynxkite-app \
  -e lynxkite-graph-analytics \
  -e lynxkite-bio \
  -e lynxkite-pillow-example
WORKDIR $HOME/app/examples
ENV PORT=7860
CMD ["uv", "run", "lynxkite"]
