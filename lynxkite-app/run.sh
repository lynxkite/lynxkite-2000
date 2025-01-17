#!/bin/bash -xue
export NX_CUGRAPH_AUTOCONFIG=True
uvicorn server.main:app --reload
