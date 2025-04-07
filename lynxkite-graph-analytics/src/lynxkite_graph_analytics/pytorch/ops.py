"""Boxes for defining PyTorch models."""

import enum
from lynxkite.core import ops
from lynxkite.core.ops import Parameter as P
import torch
import torch_geometric.nn as pyg_nn
from .core import op, reg, ENV

reg("Input: tensor", outputs=["output"], params=[P.basic("name")])
reg("Input: graph edges", outputs=["edges"])
reg("Input: sequential", outputs=["y"])
reg("Output", inputs=["x"], outputs=["x"], params=[P.basic("name")])


@op("LSTM")
def lstm(x, *, input_size=1024, hidden_size=1024, dropout=0.0):
    return torch.nn.LSTM(input_size, hidden_size, dropout=0.0)


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


@op("Attention", outputs=["outputs", "weights"])
def attention(query, key, value, *, embed_dim=1024, num_heads=1, dropout=0.0):
    return torch.nn.MultiHeadAttention(embed_dim, num_heads, dropout=dropout, need_weights=True)


@op("LayerNorm", outputs=["outputs", "weights"])
def layernorm(x, *, normalized_shape=""):
    normalized_shape = [int(s.strip()) for s in normalized_shape.split(",")]
    return torch.nn.LayerNorm(normalized_shape)


@op("Dropout", outputs=["outputs", "weights"])
def dropout(x, *, p=0.0):
    return torch.nn.Dropout(p)


@op("Linear")
def linear(x, *, output_dim=1024):
    return pyg_nn.Linear(-1, output_dim)


class ActivationTypes(enum.Enum):
    ReLU = "ReLU"
    Leaky_ReLU = "Leaky ReLU"
    Tanh = "Tanh"
    Mish = "Mish"


@op("Activation")
def activation(x, *, type: ActivationTypes = ActivationTypes.ReLU):
    return getattr(torch.nn.functional, type.name.lower().replace(" ", "_"))


@op("MSE loss")
def mse_loss(x, y):
    return torch.nn.functional.mse_loss


@op("Constant vector")
def constant_vector(*, value=0, size=1):
    return lambda _: torch.full((size,), value)


@op("Softmax")
def softmax(x, *, dim=1):
    return torch.nn.Softmax(dim=dim)


@op("Concatenate")
def concatenate(a, b):
    return lambda a, b: torch.concatenate(*torch.broadcast_tensors(a, b))


reg(
    "Graph conv",
    inputs=["x", "edges"],
    outputs=["x"],
    params=[P.options("type", ["GCNConv", "GATConv", "GATv2Conv", "SAGEConv"])],
)

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
    params=[
        ops.Parameter.basic("times", 1, int),
        ops.Parameter.basic("same_weights", False, bool),
    ],
)

ops.register_passive_op(
    ENV,
    "Recurrent chain",
    inputs=[ops.Input(name="input", position="top", type="tensor")],
    outputs=[ops.Output(name="output", position="bottom", type="tensor")],
    params=[],
)


def _set_handle_positions(op):
    op: ops.Op = op.__op__
    for v in op.outputs.values():
        v.position = "top"
    for v in op.inputs.values():
        v.position = "bottom"


def _register_simple_pytorch_layer(func):
    op = ops.op(ENV, func.__name__.title())(lambda input: func)
    _set_handle_positions(op)


def _register_two_tensor_function(func):
    op = ops.op(ENV, func.__name__.title())(lambda a, b: func)
    _set_handle_positions(op)


SIMPLE_FUNCTIONS = [
    torch.sin,
    torch.cos,
    torch.log,
    torch.exp,
]
TWO_TENSOR_FUNCTIONS = [
    torch.multiply,
    torch.add,
    torch.subtract,
]


for f in SIMPLE_FUNCTIONS:
    _register_simple_pytorch_layer(f)
for f in TWO_TENSOR_FUNCTIONS:
    _register_two_tensor_function(f)
