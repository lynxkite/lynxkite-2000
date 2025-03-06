# LynxKite Graph Analytics

This is a LynxKite plugin that provides a graph analytics environment.
Includes a batch executor operating on graph bundles and a collection of GPU-accelerated graph data science operations.

To use this, install both LynxKite and this plugin.
Then "LynxKite Graph Analytics" will show up as one of the workspace types in LynxKite.

```bash
pip install lynxkite lynxkite-graph-analytics
```

Run LynxKite with `NX_CUGRAPH_AUTOCONFIG=True` to enable GPU-accelerated graph data science operations.


## BioNemo

If you want to use BioNemo operations, then you will have to use the provided Docker image, or
install BioNemo manually in your environment.
Take into account that BioNemo needs a GPU to work, you can find the specific requirements
[here](https://docs.nvidia.com/bionemo-framework/latest/user-guide/getting-started/pre-reqs/).

The import of BioNemo operations is gate keeped behing the `LYNXKITE_BIONEMO_INSTALLED` variable.
BioNemo operations will only be imported if this environment variable is set to true.

To build the image:

```bash
# in lynxkite-graph-analytics folder
$ docker build -f Dockerfile.bionemo -t lynxkite-bionemo ..
```

Take into account that this Dockerfile does not include the lynxkite-lynxscribe package. If you want to include it you will
need to set up git credentials inside the container.

Then, inside the image you can start LynxKite as usual.

If you want to do some development, then it is recommend to use the [devcontainers](https://code.visualstudio.com/docs/devcontainers/containers)
vscode extension. The following is a basic configuration to get started:

```json
// .devcontainer/devcontainer.json
{
	"name": "Existing Dockerfile",
	"runArgs": [
		"--gpus=all",
		"--shm-size=4g"
	],
	"build": {
		"context": "..",
		"dockerfile": "../lynxkite-graph-analytics/Dockerfile.bionemo"
	}
}
```
