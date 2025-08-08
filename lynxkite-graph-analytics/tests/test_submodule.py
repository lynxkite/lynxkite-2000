import pytest
import torch
import torch_geometric.nn as pyg_nn
from lynxkite_graph_analytics.pytorch import pytorch_core
from lynxkite_core import workspace


def make_ws(env, nodes: dict[str, dict], edges: list[tuple[str, str]]):
    ws = workspace.Workspace(env=env)
    for id, data in nodes.items():
        title = data["title"]
        del data["title"]
        ws.add_node(
            id=id,
            title=title,
            params=data,
        )
    ws.edges = [
        workspace.WorkspaceEdge(
            id=f"{source}->{target}",
            source=source.split(":")[0],
            target=target.split(":")[0],
            sourceHandle=source.split(":")[1],
            targetHandle=target.split(":")[1],
        )
        for source, target in edges
    ]
    return ws


def test_identify_submodules_no_submodules():
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "n1": {"title": "Linear"},
            "n2": {"title": "Linear"},
            "opt": {"title": "Optimizer"},
        },
        [
            ("input:output", "n1:x"),
            ("n1:output", "n2:x"),
            ("n2:output", "opt:loss"),
        ],
    )
    builder = pytorch_core.ModelBuilder(ws)
    assert builder.identify_submodules() == {}


def test_identify_submodules_single_chain():
    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module):
        return lambda x: sub(x)

    # input -> n1 -> n2 -> n3(submodule input)
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "n1": {"title": "Linear"},
            "n2": {"title": "Linear"},
            "n3": {"title": "WithSubmodule"},
            "opt": {"title": "Optimizer"},
        },
        [
            ("input:output", "n1:x"),
            ("n1:output", "n2:x"),
            ("n2:output", "n3:sub"),
            ("n3:output", "opt:loss"),
        ],
    )
    builder = pytorch_core.ModelBuilder(ws)
    submodules = builder.identify_submodules()
    key = pytorch_core._to_id("n2", "output")
    assert set(submodules.keys()) == {key}
    # The subtree should be n1, n2
    assert submodules[key] == {"input", "n1", "n2"}
    submodel = builder.build_submodel(submodules[key])
    assert isinstance(submodel.module, pyg_nn.Sequential)
    assert len(submodel.module) == 2
    assert all(isinstance(layer, pyg_nn.Linear) for layer in submodel.module)


def test_identify_submodules_branching():
    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module, x):
        return lambda y: sub(y) + x

    # n1 -> n2, n2 -> n4(submodule input), n3 -> n4(non submodule input)
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "input2": {"title": "Input: tensor"},
            "n1": {"title": "Linear"},
            "n2": {"title": "Linear"},
            "n3": {"title": "Linear"},
            "n4": {"title": "WithSubmodule"},
            "opt": {"title": "Optimizer"},
        },
        [
            ("input:output", "n1:x"),
            ("n1:output", "n2:x"),
            ("n2:output", "n4:sub"),
            ("input2:output", "n3:x"),
            ("n3:output", "n4:x"),
            ("n4:output", "opt:loss"),
        ],
    )
    builder = pytorch_core.ModelBuilder(ws)
    submodules = builder.identify_submodules()
    key = pytorch_core._to_id("n2", "output")
    # The subtree for sub should be n1, n2
    assert submodules[key] == {"input", "n1", "n2"}


def test_identify_submodules_multiple_submodules():
    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module, x):
        return lambda y: sub(y) + x

    # n1 -> n2 (WithSubmodule.sub), n3 -> n2 (WithSubmodule.x)
    # n4 -> n5 (WithSubmodules.subs), n6 -> n5 (WithSubmodules.x)
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "input2": {"title": "Input: tensor"},
            "input3": {"title": "Input: tensor"},
            "n1": {"title": "Linear"},
            "n2": {"title": "WithSubmodule"},
            "n3": {"title": "Linear"},
            "n4": {"title": "Linear"},
            "n5": {"title": "WithSubmodule"},
            "n6": {"title": "Linear"},
            "opt": {"title": "Optimizer"},
        },
        [
            ("input:output", "n1:x"),
            ("n1:output", "n2:sub"),
            ("n3:output", "n2:x"),
            ("n2:output", "opt:loss"),
            ("input2:output", "n4:x"),
            ("n4:output", "n5:sub"),
            ("input3:output", "n6:x"),
            ("n6:output", "n5:x"),
            ("n5:output", "opt:loss"),
        ],
    )
    builder = pytorch_core.ModelBuilder(ws)
    submodules = builder.identify_submodules()
    key1 = pytorch_core._to_id("n1", "output")
    key2 = pytorch_core._to_id("n4", "output")
    assert set(submodules.keys()) == {key1, key2}
    assert submodules[key1] == {"input", "n1"}
    assert submodules[key2] == {"input2", "n4"}
    submodel1 = builder.build_submodel(submodules[key1]).module
    submodel2 = builder.build_submodel(submodules[key2]).module
    assert isinstance(submodel1, pyg_nn.Sequential)
    assert isinstance(submodel2, pyg_nn.Sequential)
    assert len(submodel1) == 1
    assert len(submodel2) == 1
    assert isinstance(submodel1[0], pyg_nn.Linear)
    assert isinstance(submodel2[0], pyg_nn.Linear)


def test_identify_submodules_overlapping_subtrees():
    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module):
        return lambda y: sub(y)

    # n1 -> n2 -> n3 (WithSubmodule.sub), n1 -> n4 (WithSubmodules.sub)
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "n1": {"title": "Linear"},
            "n2": {"title": "Linear"},
            "n3": {"title": "WithSubmodule"},
            "n4": {"title": "WithSubmodule"},
            "opt": {"title": "Optimizer"},
        },
        [
            ("input:output", "n1:x"),
            ("n1:output", "n2:x"),
            ("n2:output", "n3:sub"),
            ("n3:output", "opt:loss"),
            ("n1:output", "n4:sub"),
            ("n4:output", "opt:loss"),
        ],
    )
    builder = pytorch_core.ModelBuilder(ws)
    submodules = builder.identify_submodules()
    key1 = pytorch_core._to_id("n2", "output")
    key2 = pytorch_core._to_id("n1", "output")
    assert set(submodules.keys()) == {key1, key2}
    assert submodules[key1] == {"input", "n1", "n2"}
    assert submodules[key2] == {"input", "n1"}


async def test_build_model_w_submodules():
    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module):
        return sub

    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "lin1": {"title": "Linear", "output_dim": 4},
            "lin2": {"title": "Linear", "output_dim": 4},
            "lin3": {"title": "Linear", "output_dim": 4},
            "sub": {"title": "WithSubmodule"},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin1:x"),
            ("lin1:output", "lin2:x"),
            ("lin2:output", "lin3:x"),
            ("lin3:output", "sub:sub"),
            ("sub:output", "act:x"),
            ("act:output", "output:x"),
            ("output:x", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    x = torch.rand(100, 4)
    y = x + 1
    m = pytorch_core.build_model(ws)
    for i in range(1000):
        loss = m.train({"input_output": x, "label_output": y})
    assert loss < 0.1
    o = m.inference({"input_output": x[:1]})
    error = torch.nn.functional.mse_loss(o["output_x"], x[:1] + 1)
    assert error < 0.1


async def test_submodel_with_outgoing_connections_not_accepted():
    # If a submodule is connected to nodes outside its subtree,
    # and the destination node is not a submodule we should raise an error

    @pytorch_core.op("WithSubmodule")
    def with_submodule(sub: torch.nn.Module, x):
        return sub

    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "lin1": {"title": "Linear", "output_dim": 4},
            "lin2": {"title": "Linear", "output_dim": 4},
            "sub": {"title": "WithSubmodule"},
            "concatenate": {"title": "Concatenate"},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin1:x"),
            ("lin1:output", "lin2:x"),
            ("lin1:output", "concatenate:a"),
            ("lin2:output", "concatenate:b"),
            ("lin2:output", "sub:sub"),
            ("concatenate:output", "sub:x"),
            ("sub:output", "act:x"),
            ("act:output", "output:x"),
            ("output:x", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    with pytest.raises(
        ValueError,
        match="Submodule lin2_output is not valid: it has connections outside the submodule tree.",
    ):
        pytorch_core.build_model(ws)


async def test_submodel_with_outgoing_connections_to_submodule_accepted():
    # If a submodule is connected to nodes outside its subtree,
    # and the destination node is not a submodule we should raise an error

    @pytorch_core.op("WithSubmodule")
    def with_submodule(first: torch.nn.Module, second: torch.nn.Module):
        return first

    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "lin1": {"title": "Linear", "output_dim": 4},
            "lin2": {"title": "Linear", "output_dim": 4},
            "sub": {"title": "WithSubmodule"},
            "sub2": {"title": "WithSubmodule"},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin1:x"),
            ("lin1:output", "lin2:x"),
            ("lin1:output", "sub2:first"),
            ("lin2:output", "sub:first"),
            ("lin2:output", "sub2:second"),
            ("sub:output", "act:x"),
            ("sub2:output", "sub:second"),
            ("act:output", "output:x"),
            ("output:x", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    m = pytorch_core.build_model(ws)
    assert m


async def test_submodel_list_input():
    # If a submodule is connected to nodes outside its subtree,
    # and the destination node is not a submodule we should raise an error

    @pytorch_core.op("WithSubmodule")
    def with_submodule(modules: list[torch.nn.Module]):
        return modules[0]

    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "input2": {"title": "Input: tensor"},
            "lin1": {"title": "Linear", "output_dim": 4},
            "lin2": {"title": "Linear", "output_dim": 4},
            "sub": {"title": "WithSubmodule"},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin1:x"),
            ("input2:output", "lin2:x"),
            ("lin1:output", "sub:modules"),
            ("lin2:output", "sub:modules"),
            ("sub:output", "act:x"),
            ("act:output", "output:x"),
            ("output:x", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    m = pytorch_core.build_model(ws)
    assert m
