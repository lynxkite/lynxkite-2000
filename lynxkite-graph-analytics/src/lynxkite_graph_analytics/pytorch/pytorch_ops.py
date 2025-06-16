"""Boxes for defining PyTorch models."""

import enum
from lynxkite.core import ops
from lynxkite.core.ops import Parameter as P
import torch
from .pytorch_core import op, reg, ENV


class ActivationTypes(str, enum.Enum):
    ELU = "ELU"
    GELU = "GELU"
    LeakyReLU = "Leaky ReLU"
    Mish = "Mish"
    PReLU = "PReLU"
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"
    SiLU = "SiLU"
    Softplus = "Softplus"
    Tanh = "Tanh"

    def to_layer(self):
        return getattr(torch.nn, self.name.replace(" ", ""))()


class ODEMethod(str, enum.Enum):
    dopri8 = "dopri8"
    dopri5 = "dopri5"
    bosh3 = "bosh3"
    fehlberg2 = "fehlberg2"
    adaptive_heun = "adaptive_heun"
    euler = "euler"
    midpoint = "midpoint"
    rk4 = "rk4"
    explicit_adams = "explicit_adams"
    implicit_adams = "implicit_adams"


reg("Input: tensor", outputs=["output"], params=[P.basic("name")], color="gray")
reg("Input: graph edges", outputs=["edges"], params=[P.basic("name")], color="gray")
reg("Input: sequential", outputs=["y"], params=[P.basic("name")], color="gray")
reg("Output", inputs=["x"], outputs=["x"], params=[P.basic("name")], color="gray")


@op("LSTM", weights=True)
def lstm(x, *, input_size=1024, hidden_size=1024, dropout=0.0):
    lstm = torch.nn.LSTM(input_size, hidden_size, dropout=dropout, batch_first=True)
    if input_size == 1:
        return lambda x: lstm(x.unsqueeze(-1))[1][0].squeeze(0)
    return lambda x: lstm(x)[1][0].squeeze(0)


class ODEFunc(torch.nn.Module):
    def __init__(self, *, input_dim, hidden_dim, num_layers, activation_type):
        super().__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(activation_type.to_layer())
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, t, y):
        return self.mlp(y)


class ODEWithMLP(torch.nn.Module):
    def __init__(self, *, rtol, atol, input_dim, hidden_dim, num_layers, activation_type, method):
        super().__init__()
        self.func = ODEFunc(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_type=activation_type,
        )
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, state0, times):
        import torchdiffeq

        sol = torchdiffeq.odeint_adjoint(
            self.func,
            state0,
            times,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method.value,
        )
        return sol[..., 0].squeeze(-1)


@op("Neural ODE with MLP", weights=True)
def neural_ode_mlp(
    state_0,
    timestamps,
    *,
    method=ODEMethod.dopri5,
    relative_tolerance=1e-3,
    absolute_tolerance=1e-3,
    state_dimensions=1,
    mlp_layers=3,
    mlp_hidden_size=64,
    mlp_activation=ActivationTypes.ReLU,
):
    return ODEWithMLP(
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        input_dim=state_dimensions,
        hidden_dim=mlp_hidden_size,
        num_layers=mlp_layers,
        activation_type=mlp_activation,
        method=method,
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


@op("Linear", weights=True)
def linear(x, *, output_dim=1024):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.Linear(-1, output_dim)


@op("Mean pool")
def mean_pool(x):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.global_mean_pool


@op("Activation")
def activation(x, *, type: ActivationTypes = ActivationTypes.ReLU):
    return type.to_layer()


@op("MSE loss")
def mse_loss(x, y):
    return torch.nn.functional.mse_loss


@op("Constant vector")
def constant_vector(*, value=0, size=1):
    return lambda _: torch.full((size,), value)


@op("Softmax")
def softmax(x, *, dim=1):
    return torch.nn.Softmax(dim=dim)


@op("Embedding", weights=True)
def embedding(x, *, num_embeddings: int, embedding_dim: int):
    return torch.nn.Embedding(num_embeddings, embedding_dim)


def cat(a, b):
    while len(a.shape) < len(b.shape):
        a = a.unsqueeze(-1)
    while len(b.shape) < len(a.shape):
        b = b.unsqueeze(-1)
    return torch.concatenate((a, b), -1)


@op("Concatenate")
def concatenate(a, b):
    return cat


reg(
    "Pick element by index",
    inputs=["x", "index"],
    outputs=["x_i"],
)
reg(
    "Pick element by constant",
    inputs=["x"],
    outputs=["x_i"],
    params=[ops.Parameter.basic("index", "0")],
)
reg(
    "Take first n",
    inputs=["x"],
    outputs=["x"],
    params=[ops.Parameter.basic("n", 1, int)],
)
reg(
    "Drop first n",
    inputs=["x"],
    outputs=["x"],
    params=[ops.Parameter.basic("n", 1, int)],
)
reg(
    "Graph conv",
    color="blue",
    inputs=["x", "edges"],
    outputs=["x"],
    params=[P.options("type", ["GCNConv", "GATConv", "GATv2Conv", "SAGEConv"])],
)
reg(
    "Heterogeneous graph conv",
    inputs=["node_embeddings", "edge_modules"],
    outputs=["x"],
    params=[
        ops.Parameter.basic("node_embeddings_order"),
        ops.Parameter.basic("edge_modules_order"),
    ],
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
        P.basic("lr", 0.0001),
    ],
    color="green",
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
    for v in op.outputs:
        v.position = ops.Position.TOP
    for v in op.inputs:
        v.position = ops.Position.BOTTOM


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
