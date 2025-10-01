"""Boxes for defining PyTorch models."""

import enum
from lynxkite_core import ops
from lynxkite_core.ops import Parameter as P
import torch
from .pytorch_core import op, reg, ENV, input_op, InputContext
from .. import core


class ActivationTypes(enum.StrEnum):
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


class ODEMethod(enum.StrEnum):
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


class TorchTypes(enum.StrEnum):
    float = "float"
    double = "double"
    int = "int"
    long = "long"
    bool = "bool"

    def to_dtype(self):
        return getattr(torch, self.value)


@input_op("tensor")
def tensor_input(*, type: TorchTypes = TorchTypes.float, per_sample: bool = True):
    """An input tensor.

    Args:
        type: The data type of the tensor.
        per_sample: Whether this has a different value for each sample, or is constant across the dataset.
    """

    def from_bundle(
        b: core.Bundle,
        ctx: InputContext,
        *,
        table_name: core.TableName = "",
        column_name: core.ColumnNameByTableName = "",
    ):
        """
        Args:
            table_name: One column of this table will be used as input.
            column_name: The name of the column to use as input.
        """
        df = b.dfs[table_name][column_name]
        batch = ctx.batch_df(df) if per_sample else df
        t = torch.tensor(batch.to_list(), dtype=type.to_dtype())
        return t

    return from_bundle


@input_op("graph edges")
def graph_edges_input():
    """The edges of a graph as input. A 2xE tensor of src/dst indices. Not batched."""

    def from_bundle(
        b: core.Bundle,
        ctx: InputContext,
        *,
        table_name: core.TableName = "",
        source_column_name: core.ColumnNameByTableName = "",
        target_column_name: core.ColumnNameByTableName = "",
    ):
        """
        Args:
            table_name: The table with the edges.
            source_column_name: The column with source node indices.
            target_column_name: The column with target node indices.
        """
        src = b.dfs[table_name][source_column_name]
        dst = b.dfs[table_name][target_column_name]
        return torch.tensor([src, dst], dtype=torch.long)

    return from_bundle


@input_op("sequential")
def sequential_input(*, type: TorchTypes = TorchTypes.float, per_sample: bool = True):
    """An input tensor with a sequence for each sample.

    Args:
        type: The data type of the tensor.
        per_sample: Whether this has a different value for each sample, or is constant across the dataset.
    """

    def from_bundle(
        b: core.Bundle,
        ctx: InputContext,
        *,
        table_name: core.TableName = "",
        column_name: core.ColumnNameByTableName = "",
    ):
        """
        Args:
            table_name: One column of this table will be used as input.
            column_name: The name of the column to use as input.
        """
        df = b.dfs[table_name][column_name]
        batch = ctx.batch_df(df) if per_sample else df
        t = torch.tensor(batch.to_list(), dtype=type.to_dtype())
        return t

    return from_bundle


reg("Output", inputs=["x"], outputs=["x"], params=[P.basic("name")], color="gray")


@op("LSTM", weights=True)
def lstm(x, *, input_size=1024, hidden_size=1024, dropout=0.0):
    lstm = torch.nn.LSTM(input_size, hidden_size, dropout=dropout, batch_first=True)
    if input_size == 1:
        return lambda x: lstm(x.unsqueeze(-1))[1][0].squeeze(0)
    return lambda x: lstm(x)[1][0].squeeze(0)


class MLPODEFunc(torch.nn.Module):
    def __init__(self, *, input_dim, hidden_dim, output_dim, num_layers, activation_type):
        super().__init__()
        assert num_layers >= 2, "must have at least 2 layers for MLP"
        assert output_dim <= input_dim, "output dim must be <= input dim"
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            layers.append(activation_type.to_layer())
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_type.to_layer())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, t, y):
        res = self.mlp(y)
        return torch.nn.functional.pad(res, (0, self.input_dim - self.output_dim), "constant", 0.0)


class ODEWithMLP(torch.nn.Module):
    def __init__(self, *, rtol, atol, input_dim, hidden_dim, num_layers, activation_type, method):
        super().__init__()
        self.func = MLPODEFunc(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation_type=activation_type,
        )
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def forward(self, state0, times):
        import torchdiffeq

        assert state0.shape[0] == 1, "Batch size must be 1 for ODE solver."
        # Squeeze and unsqueeze for the 1-element batch.
        state0 = state0.squeeze(0)
        times = times.squeeze(0)
        sol = torchdiffeq.odeint_adjoint(
            self.func,
            state0,
            times,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method.value,
        )
        return sol[..., 0].unsqueeze(0)


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
    """A neural ODE for predicting a 1-dimensional value over time, using an MLP to model the derivative.

    Must be used with batch size 1.
    """
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
    return torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)


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
    inputs=[ops.Input(name="input", position=ops.Position.TOP, type="tensor")],
    outputs=[ops.Output(name="output", position=ops.Position.BOTTOM, type="tensor")],
    params=[
        ops.Parameter.basic("times", 1, int),
        ops.Parameter.basic("same_weights", False, bool),
    ],
)

ops.register_passive_op(
    ENV,
    "Recurrent chain",
    inputs=[ops.Input(name="input", position=ops.Position.TOP, type="tensor")],
    outputs=[ops.Output(name="output", position=ops.Position.BOTTOM, type="tensor")],
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
