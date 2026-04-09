#!/bin/bash
set -xue

# Install uv.
if ! command -v uv &> /dev/null; then
  curl -sSL https://astral.sh/uv/install.sh -o uv-installer.sh
  sh uv-installer.sh && rm uv-installer.sh
fi
uv sync --inexact

# Start the backend
cd examples
uv run lynxkite &

# Start the frontend
cd ../lynxkite-app/web
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &

# Wait for both processes
wait
