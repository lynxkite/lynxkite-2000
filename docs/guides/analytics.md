# Graph analytics & data science

Install LynxKite with the graph analytics package:

```bash
pip install lynxkite lynxkite-graph-analytics
```

Run LynxKite in your data directory:

```bash
cd lynxkite-data
lynxkite
```

LynxKite by default runs on port 8000, so you can access it in your browser at
[http://localhost:8000](http://localhost:8000).
To run it on a different port, set the `PORT` environment variable (e.g., `PORT=8080 lynxkite`).

## Using a CUDA GPU

To make full use of your GPU, install the `lynxkite-graph-analytics` package with GPU support.

```bash
pip install lynxkite 'lynxkite-graph-analytics[gpu]'
```

And start it with the cuGraph backend for NetworkX:

```bash
NX_CUGRAPH_AUTOCONFIG=true lynxkite
```

## Directory browser

When you open the LynxKite web interface, you arrive in the directory browser. You see
the files and directories in your data directory — if you just created it, it will be empty.

You can create workspaces, [code files](plugins.md), and folders in the directory browser.

## Workspaces

A LynxKite workspace is the place where you build a data science pipeline.
Pipelines are built from boxes, which have inputs and outputs that can be connected to each other.

To place a box, click anywhere in the workspace. This opens a search menu where you can
find the box you want to add.

## Importing your data

To import CSV, Parquet, JSON, and Excel files, you can simply drag and drop them into the LynxKite workspace.
This will upload the file to the server and add an "Import file" box to the workspace.

You can also create "Import file" boxes manually and type the path to the file.
You can either use an absolute path, or a relative path from the data directory.
(Where you started LynxKite.)

## Neural network design

The graph analytics package includes two environments, _"LynxKite Graph Analytics"_, and _"PyTorch model"_.
Use the dropdown in the top right corner to switch to the "PyTorch model" environment.
This environment allows you to define neural network architectures visually.

The important parts of a neural network definition are:

- **Inputs**: These boxes stand for the inputs. You will connect them to actual data in the workspace that
  uses this model.
- **Layers**: The heart of the model. Use the _"Repeat"_ box looping back from the output of a layer to the
  input of an earlier layer to repeat a set of layers.
- **Outputs**: These boxes mark the place in the data flow that holds the predictions of the model.
- **Loss**: Build the loss computation after the output box. This part is only used during training.
- **Optimizer**: The result of the loss computation goes into the optimizer. Training is partially configured
  in the optimizer box, partially in the training box in the workspace that uses the model.

Once the model is defined, you can use it in other workspaces.

- Load it with the _"Define model"_ box.
- Train it with the _"Train model"_ box.
- Generate predictions with the _"Model inference"_ box.

See the [_Model definition_ and _Model use_ workspaces](https://github.com/lynxkite/lynxkite-2000/tree/main/examples)
for a practical example.
