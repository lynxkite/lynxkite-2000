"""Infrastructure for defining PyTorch models."""

import copy
import graphlib
import typing

import pydantic
from lynxkite.core import ops, workspace
import torch
import dataclasses
from .. import core

ENV = "PyTorch model"


def op(name, weights=False, **kwargs):
    if weights:
        kwargs["color"] = "blue"
    _op = ops.op(ENV, name, **kwargs)

    def decorator(func):
        _op(func)
        op = func.__op__
        for p in op.inputs:
            p.position = ops.Position.BOTTOM
        for p in op.outputs:
            p.position = ops.Position.TOP
        return func

    return decorator


def reg(name, inputs=[], outputs=None, params=[], **kwargs):
    if outputs is None:
        outputs = inputs
    return ops.register_passive_op(
        ENV,
        name,
        inputs=[ops.Input(name=name, position="bottom", type="tensor") for name in inputs],
        outputs=[ops.Output(name=name, position="top", type="tensor") for name in outputs],
        params=params,
        **kwargs,
    )


def _to_id(*strings: str) -> str:
    """Replaces all non-alphanumeric characters with underscores."""
    return "_".join("".join(c if c.isalnum() else "_" for c in s) for s in strings)


class Layer:
    """Temporary data structure used by ModelBuilder."""

    def __init__(
        self,
        module: torch.nn.Module,
        origin_id: str,
        inputs: list[str | list[str]],
        outputs: list[str],
    ):
        self.module = module
        self.origin_id = origin_id
        self.inputs = self.flatten_inputs(inputs)
        self.outputs = outputs

    @staticmethod
    def flatten_inputs(inputs: list[str | list[str]]) -> list[str]:
        """Flattens the input list, since some inputs can be lists.
        We need to flatten them to make sure the signature for pyg.nn.Sequential. is correct.
        For example, if the inputs are ['a', ['b', 'c']], we want to return ['a', 'b', 'c'].
        """
        inputs_flat = []
        for input_item in inputs:
            if isinstance(input_item, (list, tuple)):
                inputs_flat.extend(input_item)
            else:
                inputs_flat.append(input_item)
        return inputs_flat

    def for_sequential(self):
        """The layer signature for pyg.nn.Sequential."""
        # "nothing" is used as a bogus input if an operation has no inputs.
        # The module in turn needs to take one argument, but it will always be None.
        inputs = ", ".join(self.inputs) or "nothing"
        outputs = ", ".join(self.outputs)
        return self.module, f"{inputs} -> {outputs}"


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
    input_output_names: list[str]
    loss: torch.nn.Module
    optimizer_parameters: dict[str, any]
    optimizer: torch.optim.Optimizer | None = None
    source_workspace: str | None = None
    trained: bool = False

    def __post_init__(self):
        self._make_optimizer()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.model(nothing=None, **inputs)
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
        loss = self.loss(nothing=None, **values)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _make_optimizer(self):
        # We need to make a new optimizer when the model is copied. (It's tied to its parameters.)
        p = self.optimizer_parameters
        o = getattr(torch.optim, p["type"].name)
        self.optimizer = o(self.model.parameters(), lr=p["lr"])

    def copy(self):
        """Returns a copy of the model."""
        c = dataclasses.replace(
            self,
            model=copy.deepcopy(self.model),
        )
        c._make_optimizer()
        c.optimizer.load_state_dict(self.optimizer.state_dict())
        return c

    def metadata(self):
        return {
            "type": "model",
            "model": {
                "model_inputs": self.model_inputs,
                "model_outputs": self.model_outputs,
                "loss_inputs": self.loss_inputs,
                "input_output_names": self.input_output_names,
                "trained": self.trained,
            },
        }


def build_model(ws: workspace.Workspace) -> ModelConfig:
    """Builds the model described in the workspace."""
    builder = ModelBuilder(ws)
    return builder.build_model()


class ModelBuilder:
    """The state shared between methods that are used to build the model."""

    def __init__(self, ws: workspace.Workspace):
        self.catalog = ops.CATALOGS[ENV]
        optimizers = []
        self.nodes: dict[str, workspace.WorkspaceNode] = {}
        repeats: list[str] = []
        for node in ws.nodes:
            self.nodes[node.id] = node
            if node.data.title == "Optimizer":
                optimizers.append(node.id)
            elif node.data.title == "Repeat":
                repeats.append(node.id)
                self.nodes[f"START {node.id}"] = node
                self.nodes[f"END {node.id}"] = node
        assert optimizers, "No optimizer found."
        assert len(optimizers) == 1, f"More than one optimizer found: {optimizers}"
        [self.optimizer] = optimizers
        self.dependencies = {n: [] for n in self.nodes}
        self.in_edges: dict[str, dict[str, list[tuple[str, str]]]] = {n: {} for n in self.nodes}
        self.out_edges: dict[str, dict[str, list[tuple[str, str]]]] = {n: {} for n in self.nodes}
        for e in ws.edges:
            self.dependencies[e.target].append(e.source)
            self.in_edges.setdefault(e.target, {}).setdefault(e.targetHandle, []).append(
                (e.source, e.sourceHandle)
            )
            self.out_edges.setdefault(e.source, {}).setdefault(e.sourceHandle, []).append(
                (e.target, e.targetHandle)
            )
        # Split repeat boxes into start and end, and insert them into the flow.
        # TODO: Think about recursive repeats.
        for repeat in repeats:
            if not self.out_edges[repeat] or not self.in_edges[repeat]:
                continue
            start_id = f"START {repeat}"
            end_id = f"END {repeat}"
            # repeat -> first <- real_input
            # ...becomes...
            # real_input -> start -> first
            first, firsth = self.out_edges[repeat]["output"][0]
            [(real_input, real_inputh)] = [
                k for k in self.in_edges[first][firsth] if k != (repeat, "output")
            ]
            self.dependencies[first].remove(repeat)
            self.dependencies[first].append(start_id)
            self.dependencies[start_id] = [real_input]
            self.out_edges[real_input][real_inputh] = [
                k if k != (first, firsth) else (start_id, "input")
                for k in self.out_edges[real_input][real_inputh]
            ]
            self.in_edges[start_id] = {"input": [(real_input, real_inputh)]}
            self.out_edges[start_id] = {"output": [(first, firsth)]}
            self.in_edges[first][firsth] = [(start_id, "output")]
            # repeat <- last -> real_output
            # ...becomes...
            # last -> end -> real_output
            [(last, lasth)] = self.in_edges[repeat]["input"]
            del self.dependencies[repeat]
            self.dependencies[end_id] = [last]
            real_edges = [e for e in self.out_edges[last][lasth] if e != (repeat, "input")]
            self.out_edges[last][lasth] = [(end_id, "input")]
            self.in_edges[end_id] = {"input": [(last, lasth)]}
            self.out_edges[end_id] = {"output": []}  # Populated below.
            for real_output, real_outputh in real_edges:
                self.dependencies[real_output].append(end_id)
                self.in_edges[real_output][real_outputh] = [
                    k if k != (last, lasth) else (end_id, "output")
                    for k in self.in_edges[real_output][real_outputh]
                ]
                self.out_edges[end_id]["output"].append((real_output, real_outputh))
        self.inv_dependencies = {n: [] for n in self.nodes}
        for k, v in self.dependencies.items():
            for i in v:
                self.inv_dependencies[i].append(k)
        self.layers: list[Layer] = []
        self.submodules: dict[str, torch.nn.Module] = {}
        # Clean up disconnected nodes.
        to_delete = set()
        for node_id in self.nodes:
            title = self.nodes[node_id].data.title
            if title not in self.catalog:  # Groups and comments, for example.
                to_delete.add(node_id)
                continue
            op = self.catalog[title]
            if len(self.in_edges[node_id]) != len(op.inputs):  # Unconnected inputs.
                to_delete.add(node_id)
                to_delete |= self.all_upstream(node_id)
        for node_id in to_delete:
            del self.dependencies[node_id]
            del self.in_edges[node_id]
            del self.out_edges[node_id]
            del self.inv_dependencies[node_id]
            del self.nodes[node_id]

    def all_upstream(self, node: str) -> set[str]:
        """Returns all nodes upstream of a node."""
        deps = set()
        for dep in self.dependencies[node]:
            deps.add(dep)
            deps.update(self.all_upstream(dep))
        return deps

    def all_downstream(self, node: str) -> set[str]:
        """Returns all nodes downstream of a node."""
        deps = set()
        for dep in self.inv_dependencies[node]:
            deps.add(dep)
            deps.update(self.all_downstream(dep))
        return deps

    def _is_submodule_node(self, node_id: str) -> bool:
        """Returns True if the node is a submodule

        A submodule is a node in the workspace whose output is connected to an input of another
        node, and that input is of type torch.nn.Module or list[torch.nn.Module].
        In other words, a submodule is a node that is used as a component (module) inside a
        higher-level module, rather than being a top-level layer in the model sequence.
        """
        for output, connections in self.out_edges[node_id].items():
            # TODO: What if it is connected to multiple inputs of different type?
            for target_name, target_handle in connections:
                target_node = self.nodes[target_name]
                target_op = self.catalog[target_node.data.title]
                [target_input] = [inp for inp in target_op.inputs if inp.name == target_handle]
                return self._is_submodule_type(target_input.type)
        return False

    def run_node(self, node_id: str) -> None:
        """Adds the layer(s) produced by this node to self.layers."""
        node = self.nodes[node_id]
        t = node.data.title
        op = self.catalog[t]
        p = op.convert_params(node.data.params)
        match t:
            case "Repeat":
                if node_id.startswith("END "):
                    repeat_id = node_id.removeprefix("END ")
                    start_id = f"START {repeat_id}"
                    [last_output] = self.in_edges[node_id]["input"]
                    after_start = self.all_downstream(start_id)
                    after_end = self.all_downstream(node_id)
                    before_end = self.all_upstream(node_id)
                    affected_nodes = after_start - after_end - {node_id}
                    repeated_nodes = after_start & before_end
                    assert affected_nodes == repeated_nodes, (
                        f"edges leave repeated section '{repeat_id}':\n{affected_nodes - repeated_nodes}"
                    )
                    repeated_layers = [e for e in self.layers if e.origin_id in repeated_nodes]
                    assert p["times"] >= 1, f"Cannot repeat {repeat_id} {p['times']} times."
                    for i in range(p["times"] - 1):
                        # Copy repeat section's output to repeat section's input.
                        self.layers.append(
                            Layer(
                                torch.nn.Identity(),
                                origin_id=node_id,
                                inputs=[_to_id(*last_output)],
                                outputs=[_to_id(start_id, "output")],
                            )
                        )
                        # Repeat the layers in the section.
                        for layer in repeated_layers:
                            if p["same_weights"]:
                                self.layers.append(layer)
                            else:
                                self.run_node(layer.origin_id)
                operation_result = self.run_op(node_id, op, p)
            case "Optimizer" | "Input: tensor" | "Input: graph edges" | "Input: sequential":
                return
            case _:
                operation_result = self.run_op(node_id, op, p)
        if self._is_submodule_node(node_id):
            self.submodules[node_id] = operation_result.module
        else:
            self.layers.append(operation_result)

    def _is_submodule_type(self, type_: type[typing.Any]) -> bool:
        """Returns True if the type is a submodule, i.e., it is a torch.nn.Module or list[torch.nn.Module]."""
        if type_ is torch.nn.Module or typing.get_origin(type_) is torch.nn.Module:
            return True
        if type_ is list or typing.get_origin(type_) is list:
            args = typing.get_args(type_)
            if (
                len(args) == 1
                and args[0] is torch.nn.Module
                or typing.get_origin(args[0]) is torch.nn.Module
            ):
                return True
        return False

    def _edge_to_input(self, edge: tuple[str, str], input_type: type):
        """Converts an edge to an input for an operation.

        If the input is a submodule type, it returns the submodule object.
        Otherwise, it returns the internal ID of the input.
        """
        source, source_handle = edge
        if self._is_submodule_type(input_type):
            # If the input is a submodule, we need to get the module object.
            return self.submodules[source]
        return _to_id(source, source_handle)

    def run_op(self, node_id: str, op: ops.Op, params) -> Layer:
        """Returns the layer produced by this op."""
        operation_inputs = []
        for input in op.inputs:
            input_edges = [
                self._edge_to_input(edge, input.type) for edge in self.in_edges[node_id][input.name]
            ]
            if not (input.type is list or typing.get_origin(input.type) is list):
                assert len(input_edges) == 1, (
                    f"Detected multiple input edges for non-list input {node_id} {input.name}."
                )
                [input_edges] = input_edges
            operation_inputs.append(input_edges)
        outputs = [_to_id(node_id, n.name) for n in op.outputs]
        if op.func == ops.no_op:
            module = torch.nn.Identity()
        else:
            module = op.func(*operation_inputs, **params)
        # Sub-modules are not real inputs, they are just a way to build hierarchical models.
        non_module_inputs = [
            inp
            for inp in operation_inputs
            if not (
                isinstance(inp, torch.nn.Module)
                or isinstance(inp, list)
                and all(isinstance(i, torch.nn.Module) for i in inp)
            )
        ]
        return Layer(module, node_id, non_module_inputs, outputs)

    def build_model(self) -> ModelConfig:
        # Walk the graph in topological order.
        ts = graphlib.TopologicalSorter(self.dependencies)
        for node_id in ts.static_order():
            self.run_node(node_id)
        return self.get_config()

    def get_config(self) -> ModelConfig:
        import torch_geometric.nn as pyg_nn

        # Split the design into model and loss.
        model_nodes = set()
        for node_id in self.nodes:
            if self.nodes[node_id].data.title == "Output":
                model_nodes.add(node_id)
                model_nodes |= self.all_upstream(node_id)
        assert model_nodes, "The model definition must have at least one Output node."
        layers = []
        loss_layers = []
        for layer in self.layers:
            if layer.origin_id in model_nodes:
                layers.append(layer)
            else:
                loss_layers.append(layer)
        used_in_model = set(input for layer in layers for input in layer.inputs)
        used_in_loss = set(input for layer in loss_layers for input in layer.inputs)
        made_in_model = set(output for layer in layers for output in layer.outputs)
        made_in_loss = set(output for layer in loss_layers for output in layer.outputs)
        layers = [layer.for_sequential() for layer in layers]
        loss_layers = [layer.for_sequential() for layer in loss_layers]
        cfg = {}
        cfg["model_inputs"] = list(used_in_model - made_in_model)
        cfg["model_outputs"] = list(made_in_model & used_in_loss)
        cfg["loss_inputs"] = list(used_in_loss - made_in_loss)
        cfg["input_output_names"] = self.get_names(
            *cfg["model_inputs"], *cfg["model_outputs"], *cfg["loss_inputs"]
        )
        # Make sure the trained output is output from the last model layer.
        outputs = ", ".join(cfg["model_outputs"])
        layers.append((torch.nn.Identity(), f"{outputs} -> {outputs}"))
        # Create model.
        cfg["model"] = pyg_nn.Sequential(", ".join(cfg["model_inputs"]), layers)
        # Make sure the loss is output from the last loss layer.
        [(lossb, lossh)] = self.in_edges[self.optimizer]["loss"]
        lossi = _to_id(lossb, lossh)
        loss_layers.append((torch.nn.Identity(), f"{lossi} -> loss"))
        # Create loss function.
        cfg["loss"] = pyg_nn.Sequential(", ".join(cfg["loss_inputs"]), loss_layers)
        assert not list(cfg["loss"].parameters()), f"loss should have no parameters: {loss_layers}"
        # Create optimizer.
        op = self.catalog["Optimizer"]
        cfg["optimizer_parameters"] = op.convert_params(self.nodes[self.optimizer].data.params)
        return ModelConfig(**cfg)

    def get_names(self, *ids: list[str]) -> dict[str, str]:
        """Returns a mapping from internal IDs to human-readable names."""
        names = {}
        for i in ids:
            for node in self.nodes.values():
                title = node.data.title
                op = self.catalog[title]
                name = node.data.params.get("name") or title
                for output in op.outputs:
                    i2 = _to_id(node.id, output.name)
                    if i2 == i:
                        if len(op.outputs) == 1:
                            names[i] = name
                        else:
                            names[i] = f"{name} ({output.name})"
                        break
                else:
                    continue
                break
            else:
                raise ValueError(f"Cannot find name for input {i}.")
        return names


def to_tensors(b: core.Bundle, m: ModelMapping | None) -> dict[str, torch.Tensor]:
    """Converts a tensor to the correct type for PyTorch. Ignores missing mappings."""
    if m is None:
        return {}
    tensors = {}
    for k, v in m.map.items():
        if v.df in b.dfs:
            if v.column and v.column in b.dfs[v.df]:
                tensors[k] = torch.tensor(b.dfs[v.df][v.column].to_list(), dtype=torch.float32)
            else:
                # No column specified, use the whole DataFrame.
                # TODO: Temporary hack, remove. Substitute with given the user the ability
                # to specify the type in the mapping.
                if k in [
                    "Input__tensor_2_output",
                    "Input__tensor_3_output",
                    "Input__tensor_4_output",
                    "Input__tensor_6_output",
                ]:
                    tensors[k] = torch.tensor(b.dfs[v.df].to_numpy(), dtype=torch.long)
                else:
                    tensors[k] = torch.tensor(b.dfs[v.df].to_numpy(), dtype=torch.float32)
    return tensors
