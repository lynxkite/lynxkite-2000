"""Boxes for defining PyTorch models."""

import copy
import enum
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


def op(name, **kwargs):
    _op = ops.op(ENV, name, **kwargs)

    def decorator(func):
        _op(func)
        op = func.__op__
        for p in op.inputs.values():
            p.position = "bottom"
        for p in op.outputs.values():
            p.position = "top"
        return func

    return decorator


def reg(name, inputs=[], outputs=None, params=[]):
    if outputs is None:
        outputs = inputs
    return ops.register_passive_op(
        ENV,
        name,
        inputs=[ops.Input(name=name, position="bottom", type="tensor") for name in inputs],
        outputs=[ops.Output(name=name, position="top", type="tensor") for name in outputs],
        params=params,
    )


reg("Input: tensor", outputs=["x"], params=[P.basic("name")])
reg("Input: graph edges", outputs=["edges"])
reg("Input: sequential", outputs=["y"])

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


@op("Linear")
def linear(x, *, output_dim="same"):
    if output_dim == "same":
        oshape = x.shape
    else:
        oshape = tuple(*x.shape[:-1], int(output_dim))
    return Layer(torch.nn.Linear(x.shape, oshape), shape=oshape)


class ActivationTypes(enum.Enum):
    ReLU = "ReLU"
    Leaky_ReLU = "Leaky ReLU"
    Tanh = "Tanh"
    Mish = "Mish"


@op("Activation")
def activation(x, *, type: ActivationTypes = ActivationTypes.ReLU):
    f = getattr(torch.nn.functional, type.name.lower().replace(" ", "_"))
    return Layer(f, shape=x.shape)


reg("Softmax", inputs=["x"])
reg(
    "Graph conv",
    inputs=["x", "edges"],
    outputs=["x"],
    params=[P.options("type", ["GCNConv", "GATConv", "GATv2Conv", "SAGEConv"])],
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


def _to_id(*strings: str) -> str:
    """Replaces all non-alphanumeric characters with underscores."""
    return "_".join("".join(c if c.isalnum() else "_" for c in s) for s in strings)


@dataclasses.dataclass
class OpInput:
    """Ops get their inputs like this. They have to return a Layer made for this input."""

    id: str
    shape: tuple[int, ...]


@dataclasses.dataclass
class Layer:
    """Return this from an op. Must include a module and the shapes of the outputs."""

    module: torch.nn.Module
    shapes: list[tuple[int, ...]] | None = None  # One for each output.
    shape: dataclasses.InitVar[tuple[int, ...] | None] = None  # Convenience for single output.

    def __post_init__(self, shape):
        assert not self.shapes or not shape, "Cannot set both shapes and shape."
        if shape:
            self.shapes = [shape]


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


def _add_op(op, params, inputs, outputs, sizes, layers):
    op_inputs = []
    for i in op.inputs.keys():
        id = getattr(inputs, i)
        op_inputs.append(OpInput(id, shape=sizes.get(id, 1)))
    if op.func != ops.no_op:
        layer = op.func(*op_inputs, **params)
    else:
        layer = Layer(torch.nn.Identity(), shapes=[i.shape for i in op_inputs])
    input_ids = ", ".join(i.id for i in op_inputs)
    output_ids = []
    for o, shape in zip(op.outputs.keys(), layer.shapes):
        id = getattr(outputs, o)
        output_ids.append(id)
        sizes[id] = shape
    output_ids = ", ".join(output_ids)
    layers.append((layer.module, f"{input_ids} -> {output_ids}"))


def _all_dependencies(node: str, dependencies: dict[str, list[str]]) -> set[str]:
    """Returns all dependencies of a node."""
    deps = set()
    for dep in dependencies[node]:
        deps.add(dep)
        deps.update(_all_dependencies(dep, dependencies))
    return deps


def build_model(ws: workspace.Workspace, inputs: dict[str, torch.Tensor]) -> ModelConfig:
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
    inv_dependencies = {n.id: [] for n in ws.nodes}
    in_edges = {}
    out_edges = {}
    repeats = []
    for e in ws.edges:
        if nodes[e.target].data.title == "Repeat":
            repeats.append(e.target)
        dependencies[e.target].append(e.source)
        inv_dependencies[e.source].append(e.target)
        in_edges.setdefault(e.target, {}).setdefault(e.targetHandle, []).append(
            (e.source, e.sourceHandle)
        )
        out_edges.setdefault(e.source, {}).setdefault(e.sourceHandle, []).append(
            (e.target, e.targetHandle)
        )
    # Split repeat boxes into start and end, and insert them into the flow.
    # TODO: Think about recursive repeats.
    for repeat in repeats:
        start_id = f"START {repeat}"
        end_id = f"END {repeat}"
        # repeat -> first <- real_input
        # ...becomes...
        # real_input -> start -> first
        first, firsth = out_edges[repeat]["output"][0]
        [(real_input, real_inputh)] = [
            k for k in in_edges[first][firsth] if k != (repeat, "output")
        ]
        dependencies[first].remove(repeat)
        dependencies[first].append(start_id)
        dependencies[start_id] = [real_input]
        out_edges[real_input][real_inputh] = [
            k if k != (first, firsth) else (start_id, "input")
            for k in out_edges[real_input][real_inputh]
        ]
        in_edges[start_id] = {"input": [(real_input, real_inputh)]}
        out_edges[start_id] = {"output": [(first, firsth)]}
        in_edges[first][firsth] = [(start_id, "output")]
        # repeat <- last -> real_output
        # ...becomes...
        # last -> end -> real_output
        last, lasth = in_edges[repeat]["input"][0]
        [(real_output, real_outputh)] = [
            k for k in out_edges[last][lasth] if k != (repeat, "input")
        ]
        del dependencies[repeat]
        dependencies[end_id] = [last]
        dependencies[real_output].append(end_id)
        out_edges[last][lasth] = [(end_id, "input")]
        in_edges[end_id] = {"input": [(last, lasth)]}
        out_edges[end_id] = {"output": [(real_output, real_outputh)]}
        in_edges[real_output][real_outputh] = [
            k if k != (last, lasth) else (end_id, "output")
            for k in in_edges[real_output][real_outputh]
        ]
    # Walk the graph in topological order.
    sizes = {}
    for k, i in inputs.items():
        sizes[k] = i.shape[-1]
    ts = graphlib.TopologicalSorter(dependencies)
    layers = []
    loss_layers = []
    regions: dict[str, set[str]] = {node_id: set() for node_id in dependencies}
    cfg = {}
    used_in_model = set()
    made_in_model = set()
    used_in_loss = set()
    made_in_loss = set()
    for node_id in ts.static_order():
        if node_id.startswith("START "):
            node = nodes[node_id.removeprefix("START ")]
        elif node_id.startswith("END "):
            node = nodes[node_id.removeprefix("END ")]
        else:
            node = nodes[node_id]
        t = node.data.title
        op = catalog[t]
        p = op.convert_params(node.data.params)
        for b in dependencies[node_id]:
            regions[node_id] |= regions[b]
        if "loss" in t:
            regions[node_id].add("loss")
        inputs = {}
        for n in in_edges.get(node_id, []):
            for b, h in in_edges[node_id][n]:
                i = _to_id(b, h)
                inputs[n] = i
                if "loss" in regions[node_id]:
                    used_in_loss.add(i)
                else:
                    used_in_model.add(i)
        outputs = {}
        for out in out_edges.get(node_id, []):
            i = _to_id(node_id, out)
            outputs[out] = i
            if not t.startswith("Input:"):  # The outputs of inputs are not "made" by us.
                if "loss" in regions[node_id]:
                    made_in_loss.add(i)
                else:
                    made_in_model.add(i)
        inputs = types.SimpleNamespace(**inputs)
        outputs = types.SimpleNamespace(**outputs)
        ls = loss_layers if "loss" in regions[node_id] else layers
        match t:
            case "MSE loss":
                ls.append(
                    (
                        torch.nn.functional.mse_loss,
                        f"{inputs.x}, {inputs.y} -> {outputs.loss}",
                    )
                )
            case "Repeat":
                ls.append((torch.nn.Identity(), f"{inputs.input} -> {outputs.output}"))
                sizes[outputs.output] = sizes.get(inputs.input, 1)
                if node_id.startswith("START "):
                    regions[node_id].add(("repeat", node_id.removeprefix("START ")))
                else:
                    repeat_id = node_id.removeprefix("END ")
                    start_id = f"START {repeat_id}"
                    print(f"repeat {repeat_id} ending")
                    after_start = _all_dependencies(start_id, inv_dependencies)
                    after_end = _all_dependencies(node_id, inv_dependencies)
                    before_end = _all_dependencies(node_id, dependencies)
                    affected_nodes = after_start - after_end
                    repeated_nodes = after_start & before_end
                    assert affected_nodes == repeated_nodes, (
                        f"edges leave repeated section '{repeat_id}':\n{affected_nodes - repeated_nodes}"
                    )
                    regions[node_id].remove(("repeat", repeat_id))
                    for n in repeated_nodes:
                        print(f"repeating {n}")
            case "Optimizer" | "Input: tensor" | "Input: graph edges" | "Input: sequential":
                pass
            case _:
                _add_op(op, p, inputs, outputs, sizes, ls)
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
            tensors[k] = torch.tensor(b.dfs[v.df][v.column].to_list(), dtype=torch.float32)
    return tensors
