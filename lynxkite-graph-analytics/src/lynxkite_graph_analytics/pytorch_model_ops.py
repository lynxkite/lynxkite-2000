"""Boxes for defining PyTorch models."""

import copy
import graphlib
import types

import pydantic
from lynxkite.core import ops, workspace
from lynxkite.core.ops import Parameter as P
import torch
import torch_geometric as pyg
import dataclasses
from . import core

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
    params=[P.options("type", ["ReLU", "Leaky ReLU", "Tanh", "Mish"])],
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
    params=[
        ops.Parameter.basic("times", 1, int),
        ops.Parameter.basic("same_weights", True, bool),
    ],
)

ops.register_passive_op(
    ENV,
    "Recurrent chain",
    inputs=[ops.Input(name="input", position="top", type="tensor")],
    outputs=[ops.Output(name="output", position="bottom", type="tensor")],
    params=[],
)


def _to_id(*strings: str) -> str:
    """Replaces all non-alphanumeric characters with underscores."""
    return "_".join("".join(c if c.isalnum() else "_" for c in s) for s in strings)


class ColumnSpec(pydantic.BaseModel):
    df: str
    column: str


class ModelMapping(pydantic.BaseModel):
    map: dict[str, ColumnSpec]


@dataclasses.dataclass
class ModelConfig:
    model: torch.nn.Module
    model_inputs: list[str]
    model_outputs: list[str]
    loss_inputs: list[str]
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    source_workspace: str | None = None
    trained: bool = False

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model_inputs = [inputs[i] for i in self.model_inputs]
        output = self.model(*model_inputs)
        if not isinstance(output, tuple):
            output = (output,)
        values = {k: v for k, v in zip(self.model_outputs, output)}
        return values

    def inference(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: Do multiple batches.
        self.model.eval()
        return self._forward(inputs)

    def train(self, inputs: dict[str, torch.Tensor]) -> float:
        """Train the model for one epoch. Returns the loss."""
        # TODO: Do multiple batches.
        self.model.train()
        self.optimizer.zero_grad()
        values = self._forward(inputs)
        values.update(inputs)
        loss_inputs = [values[i] for i in self.loss_inputs]
        loss = self.loss(*loss_inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def copy(self):
        """Returns a copy of the model."""
        c = dataclasses.replace(self)
        c.model = copy.deepcopy(self.model)
        return c

    def metadata(self):
        return {
            "type": "model",
            "model": {
                "inputs": self.model_inputs,
                "outputs": self.model_outputs,
                "loss_inputs": self.loss_inputs,
                "trained": self.trained,
            },
        }


def build_model(
    ws: workspace.Workspace, inputs: dict[str, torch.Tensor]
) -> ModelConfig:
    """Builds the model described in the workspace."""
    catalog = ops.CATALOGS[ENV]
    optimizers = []
    nodes = {}
    for node in ws.nodes:
        nodes[node.id] = node
        if node.data.title == "Optimizer":
            optimizers.append(node.id)
    assert optimizers, "No optimizer found."
    assert len(optimizers) == 1, f"More than one optimizer found: {optimizers}"
    [optimizer] = optimizers
    dependencies = {n.id: [] for n in ws.nodes}
    in_edges = {}
    out_edges = {}
    # TODO: Dissolve repeat boxes here.
    for e in ws.edges:
        dependencies[e.target].append(e.source)
        in_edges.setdefault(e.target, {}).setdefault(e.targetHandle, []).append(
            (e.source, e.sourceHandle)
        )
        out_edges.setdefault(e.source, {}).setdefault(e.sourceHandle, []).append(
            (e.target, e.targetHandle)
        )
    sizes = {}
    for k, i in inputs.items():
        sizes[k] = i.shape[-1]
    ts = graphlib.TopologicalSorter(dependencies)
    layers = []
    loss_layers = []
    in_loss = set()
    cfg = {}
    used_in_model = set()
    made_in_model = set()
    used_in_loss = set()
    made_in_loss = set()
    for node_id in ts.static_order():
        node = nodes[node_id]
        t = node.data.title
        op = catalog[t]
        p = op.convert_params(node.data.params)
        for b in dependencies[node_id]:
            if b in in_loss:
                in_loss.add(node_id)
        if "loss" in t:
            in_loss.add(node_id)
        inputs = {}
        for n in in_edges.get(node_id, []):
            for b, h in in_edges[node_id][n]:
                i = _to_id(b, h)
                inputs[n] = i
                if node_id in in_loss:
                    used_in_loss.add(i)
                else:
                    used_in_model.add(i)
        outputs = {}
        for out in out_edges.get(node_id, []):
            i = _to_id(node_id, out)
            outputs[out] = i
            if inputs:  # Nodes with no inputs are input nodes. Their outputs are not "made" by us.
                if node_id in in_loss:
                    made_in_loss.add(i)
                else:
                    made_in_model.add(i)
        inputs = types.SimpleNamespace(**inputs)
        outputs = types.SimpleNamespace(**outputs)
        ls = loss_layers if node_id in in_loss else layers
        match t:
            case "Linear":
                isize = sizes.get(inputs.x, 1)
                osize = isize if p["output_dim"] == "same" else int(p["output_dim"])
                ls.append((torch.nn.Linear(isize, osize), f"{inputs.x} -> {outputs.x}"))
                sizes[outputs.x] = osize
            case "Activation":
                f = getattr(
                    torch.nn.functional, p["type"].name.lower().replace(" ", "_")
                )
                ls.append((f, f"{inputs.x} -> {outputs.x}"))
                sizes[outputs.x] = sizes.get(inputs.x, 1)
            case "MSE loss":
                ls.append(
                    (
                        torch.nn.functional.mse_loss,
                        f"{inputs.x}, {inputs.y} -> {outputs.loss}",
                    )
                )
    cfg["model_inputs"] = list(used_in_model - made_in_model)
    cfg["model_outputs"] = list(made_in_model & used_in_loss)
    cfg["loss_inputs"] = list(used_in_loss - made_in_loss)
    # Make sure the trained output is output from the last model layer.
    outputs = ", ".join(cfg["model_outputs"])
    layers.append((torch.nn.Identity(), f"{outputs} -> {outputs}"))
    # Create model.
    cfg["model"] = pyg.nn.Sequential(", ".join(cfg["model_inputs"]), layers)
    # Make sure the loss is output from the last loss layer.
    [(lossb, lossh)] = in_edges[optimizer]["loss"]
    lossi = _to_id(lossb, lossh)
    loss_layers.append((torch.nn.Identity(), f"{lossi} -> loss"))
    # Create loss function.
    cfg["loss"] = pyg.nn.Sequential(", ".join(cfg["loss_inputs"]), loss_layers)
    assert not list(cfg["loss"].parameters()), (
        f"loss should have no parameters: {list(cfg['loss'].parameters())}"
    )
    # Create optimizer.
    op = catalog["Optimizer"]
    p = op.convert_params(nodes[optimizer].data.params)
    o = getattr(torch.optim, p["type"].name)
    cfg["optimizer"] = o(cfg["model"].parameters(), lr=p["lr"])
    return ModelConfig(**cfg)


def to_tensors(b: core.Bundle, m: ModelMapping | None) -> dict[str, torch.Tensor]:
    """Converts a tensor to the correct type for PyTorch. Ignores missing mappings."""
    if m is None:
        return {}
    tensors = {}
    for k, v in m.map.items():
        if v.df in b.dfs and v.column in b.dfs[v.df]:
            tensors[k] = torch.tensor(
                b.dfs[v.df][v.column].to_list(), dtype=torch.float32
            )
    return tensors
