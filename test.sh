#!/bin/bash -xue

cd "$(dirname $0)"
pytest --asyncio-mode=auto
cd lynxkite-app/web
npm run test
