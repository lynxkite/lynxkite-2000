# LynxKite Graph Analytics

This is a LynxKite plugin that provides a graph analytics environment.
Includes a batch executor operating on graph bundles and a collection of GPU-accelerated graph data science operations.

To use this, install both LynxKite and this plugin.
Then "LynxKite Graph Analytics" will show up as one of the workspace types in LynxKite.

```bash
pip install lynxkite lynxkite-graph-analytics
```

Run LynxKite with `NX_CUGRAPH_AUTOCONFIG=True` to enable GPU-accelerated graph data science operations.
