"""Boxes for defining PyTorch models."""

from lynxkite.core import ops, workspace
from lynxkite.core.ops import Parameter as P
import torch
import torch_geometric as pyg

ENV = "PyTorch model"


def reg(name, inputs=[], outputs=None, params=[]):
    if outputs is None:
        outputs = inputs
    return ops.register_passive_op(
        ENV,
        name,
        inputs=[
            ops.Input(name=name, position="bottom", type="tensor") for name in inputs
        ],
        outputs=[
            ops.Output(name=name, position="top", type="tensor") for name in outputs
        ],
        params=params,
    )


reg("Input: embedding", outputs=["x"])
reg("Input: graph edges", outputs=["edges"])
reg("Input: label", outputs=["y"])
reg("Input: positive sample", outputs=["x_pos"])
reg("Input: negative sample", outputs=["x_neg"])
reg("Input: sequential", outputs=["y"])
reg("Input: zeros", outputs=["x"])

reg("LSTM", inputs=["x", "h"], outputs=["x", "h"])
reg(
    "Neural ODE",
    inputs=["x"],
    params=[
        P.basic("relative_tolerance"),
        P.basic("absolute_tolerance"),
        P.options(
            "method",
            [
                "dopri8",
                "dopri5",
                "bosh3",
                "fehlberg2",
                "adaptive_heun",
                "euler",
                "midpoint",
                "rk4",
                "explicit_adams",
                "implicit_adams",
            ],
        ),
    ],
)
reg("Attention", inputs=["q", "k", "v"], outputs=["x", "weights"])
reg("LayerNorm", inputs=["x"])
reg("Dropout", inputs=["x"], params=[P.basic("p", 0.5)])
reg("Linear", inputs=["x"], params=[P.basic("output_dim", "same")])
reg("Softmax", inputs=["x"])
reg(
    "Graph conv",
    inputs=["x", "edges"],
    outputs=["x"],
    params=[P.options("type", ["GCNConv", "GATConv", "GATv2Conv", "SAGEConv"])],
)
reg(
    "Activation",
    inputs=["x"],
    params=[P.options("type", ["ReLU", "LeakyReLU", "Tanh", "Mish"])],
)
reg("Concatenate", inputs=["a", "b"], outputs=["x"])
reg("Add", inputs=["a", "b"], outputs=["x"])
reg("Subtract", inputs=["a", "b"], outputs=["x"])
reg("Multiply", inputs=["a", "b"], outputs=["x"])
reg("MSE loss", inputs=["x", "y"], outputs=["loss"])
reg("Triplet margin loss", inputs=["x", "x_pos", "x_neg"], outputs=["loss"])
reg("Cross-entropy loss", inputs=["x", "y"], outputs=["loss"])
reg(
    "Optimizer",
    inputs=["loss"],
    outputs=[],
    params=[
        P.options(
            "type",
            [
                "AdamW",
                "Adafactor",
                "Adagrad",
                "SGD",
                "Lion",
                "Paged AdamW",
                "Galore AdamW",
            ],
        ),
        P.basic("lr", 0.001),
    ],
)

ops.register_passive_op(
    ENV,
    "Repeat",
    inputs=[ops.Input(name="input", position="top", type="tensor")],
    outputs=[ops.Output(name="output", position="bottom", type="tensor")],
    params=[ops.Parameter.basic("times", 1, int)],
)

ops.register_passive_op(
    ENV,
    "Recurrent chain",
    inputs=[ops.Input(name="input", position="top", type="tensor")],
    outputs=[ops.Output(name="output", position="bottom", type="tensor")],
    params=[],
)


def build_model(ws: workspace.Workspace, inputs: dict):
    """Builds the model described in the workspace."""
    optimizers = []
    for node in ws.nodes:
        if node.op.name == "Optimizer":
            optimizers.append(node)
    assert optimizers, "No optimizer found."
    assert len(optimizers) == 1, f"More than one optimizer found: {optimizers}"
    [optimizer] = optimizers
    inputs = {n.id: [] for n in ws.nodes}
    for e in ws.edges:
        inputs[e.target].append(e.source)
    layers = []
    # TODO: Create layers based on the workspace.
    sizes = {}
    for k, v in inputs.items():
        sizes[k] = v.size
    layers.append((pyg.nn.Linear(sizes["x"], 1024), "x -> x"))
    layers.append((torch.nn.LayerNorm(1024), "x -> x"))
    m = pyg.nn.Sequential("x, edge_index", layers)
    return m
