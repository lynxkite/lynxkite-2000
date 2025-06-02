---
title: LynxKite 2000:MM
emoji: 🪁
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

# LynxKite 2000:MM Enterprise

LynxKite 2000:MM is a GPU-accelerated data science platform and a general tool for collaboratively edited workflows.

Features include:

- A web UI for building and executing data science workflows.
- An extensive toolbox of graph analytics operations powered by NVIDIA RAPIDS (CUDA).
- An integrated collaborative code editor makes it easy to add new operations.
- An environment for visually designing neural network model architectures.
- The infrastructure for easily creating other workflow design environments. See `lynxkite-pillow-example` for a simple
  example.

This is the next evolution of the classical [LynxKite](https://github.com/lynxkite/lynxkite). The two tools offer
similar functionality, but are not compatible. This version runs on GPU clusters instead of Hadoop clusters. It targets
CUDA instead of Apache Spark. It is much more extensible.

## Installation

```bash
pip install lynxkite lynxkite-graph-analytics
```

## Getting started

- [Online demo](https://lynx-analytics-lynxkite.hf.space/)
- [Quickstart](https://lynxkite.github.io/lynxkite-2000/guides/quickstart/)
- [Contributing](https://lynxkite.github.io/lynxkite-2000/contributing/)

## License

LynxKite 2000:MM Enterprise is built on top of the open-source
[LynxKite 2000:MM](https://github.com/lynxkite/lynxkite-2000).

Inquire with [Lynx Analytics](https://www.lynxanalytics.com/) for the licensing of this repository.
