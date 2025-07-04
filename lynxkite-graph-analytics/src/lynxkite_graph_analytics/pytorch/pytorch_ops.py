"""Boxes for defining PyTorch models."""

import enum
from lynxkite.core import ops
from lynxkite.core.ops import Parameter as P
import torch
import torch_geometric.nn as pyg_nn
from .pytorch_core import op, reg, ENV

reg("Input: Bundle", outputs=["data"], params=[P.basic("name")], color="gray")
reg("Input: tensor", outputs=["output"], params=[P.basic("name")], color="gray")
reg("Input: graph edges", outputs=["edges"], params=[P.basic("name")], color="gray")
reg("Input: sequential", outputs=["y"], params=[P.basic("name")], color="gray")
reg("Output", inputs=["x"], outputs=["x"], params=[P.basic("name")], color="gray")


@op("LSTM", weights=True)
def lstm(x, *, input_size=1024, hidden_size=1024, dropout=0.0):
    return torch.nn.LSTM(input_size, hidden_size, dropout=dropout)


reg(
    "Neural ODE with MLP",
    color="blue",
    inputs=["x", "y0", "t"],
    outputs=["y"],
    params=[
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
        P.basic("relative_tolerance"),
        P.basic("absolute_tolerance"),
        P.basic("mlp_layers"),
        P.basic("mlp_hidden_size"),
        P.options("mlp_activation", ["ReLU", "Tanh", "Sigmoid"]),
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


@op("Linear", weights=True)
def linear(x, *, in_channels: int, out_channels: int):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.Linear(in_channels, out_channels)


# @op("Mean pool")
# def mean_pool(x):
#     import torch_geometric.nn as pyg_nn

#     return pyg_nn.global_mean_pool


class ActivationTypes(str, enum.Enum):
    ReLU = "ReLU"
    Leaky_ReLU = "Leaky ReLU"
    Tanh = "Tanh"
    Mish = "Mish"


@op("Activation")
def activation(x, *, type: ActivationTypes = ActivationTypes.ReLU):
    return getattr(torch.nn.functional, type.name.lower().replace(" ", "_"))


@op("Graph Convolution", weights=True)
def graph_conv(nodes, edges, *, in_channels: int, out_channels: int):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.GCNConv(in_channels, out_channels)


class BundleHeteroConv(pyg_nn.HeteroConv):
    def __init__(
        self, convs: dict, node_names: list[str], relation_names: list[tuple[str, str, str]]
    ):
        self.convs_dict = convs
        self.node_names = node_names
        self.relation_names = relation_names
        super().__init__(convs, aggr="mean")

    def forward(self, *args):
        """Forward pass of the hetero convolution layer.

        Args:
            *args: A list of tensors, where the first `len(self.node_names)` tensors
                correspond to node features and the next `len(self.relation_names)` tensors
                correspond to edge indices for the respective relations.
        """
        assert len(args) == len(self.node_names) + len(self.relation_names), (
            f"Expected {len(self.node_names)} node tensors and {len(self.relation_names)} edge tensors, "
            f"but got {len(args)} total tensors."
        )
        x_dict = {name.strip(): tensor for name, tensor in zip(self.node_names, args)}
        edge_index_dict = dict(zip(self.relation_names, args[len(x_dict) :]))
        return super().forward(x_dict=x_dict, edge_index_dict=edge_index_dict)


@op(
    "HeteroConv",
    view="hetero_conv",
    weights=True,
    outputs=["x_dict"],
)
def hetero_conv(
    nodes: list[torch.Tensor],
    edges: list[torch.Tensor],
    *,
    node_names_str: str,
    relation_names_str: str,
    layers="[]",
):
    """Returns a :class:`~torch_geometric.nn.HeteroConv` layer."""

    import json

    if isinstance(layers, str):
        config = json.loads(layers or "[]")
    else:
        config = layers

    convs = {}
    for layer in config:
        relation = tuple(layer.get("relation", []))
        conv_type = layer.get("type", "GraphConv")
        params = layer.get("params", {})
        conv_cls = getattr(pyg_nn, conv_type)
        convs[relation] = conv_cls(**params, add_self_loops=False, concat=False)
    node_names = node_names_str.split(",") if node_names_str else []
    relation_names = relation_names_str.split(",") if relation_names_str else []
    relation_names = [tuple(r.strip().split("-")) for r in relation_names]
    return BundleHeteroConv(convs, node_names, relation_names)


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


@op("Concatenate")
def concatenate(a, b):
    return lambda a, b: torch.concatenate(*torch.broadcast_tensors(a, b))


@op("Pick element by constant")
def pick_element_by_constant(x_dict: dict, *, key: str):
    """Returns the element at the specified index from the input tensor."""
    import torch.nn as nn

    class MyFunctionModule(nn.Module):
        def forward(self, x_dict):
            return x_dict.get(key)

    return MyFunctionModule()


reg(
    "Pick element by index",
    inputs=["x", "index"],
    outputs=["x_i"],
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
TWO_TENSOR_FUNCTIONS = [torch.multiply, torch.add, torch.subtract, pyg_nn.global_mean_pool]


for f in SIMPLE_FUNCTIONS:
    _register_simple_pytorch_layer(f)
for f in TWO_TENSOR_FUNCTIONS:
    _register_two_tensor_function(f)
