from lynxkite.core import workspace
from lynxkite_graph_analytics.pytorch import pytorch_core
import torch
import pytest


def make_ws(env, nodes: dict[str, dict], edges: list[tuple[str, str]]):
    ws = workspace.Workspace(env=env)
    for id, data in nodes.items():
        title = data["title"]
        del data["title"]
        ws.add_node(
            id=id,
            data=workspace.WorkspaceNodeData(title=title, params=data),
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


def summarize_layers(m: pytorch_core.ModelConfig) -> str:
    return "".join(str(e)[:2] for e in m.model)


def summarize_connections(m: pytorch_core.ModelConfig) -> str:
    return " ".join(
        "".join(n[0] for n in c.param_names) + "->" + "".join(n[0] for n in c.return_names)
        for c in m.model._children
    )


async def test_build_model():
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "lin": {"title": "Linear", "output_dim": 4},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin:x"),
            ("lin:output", "act:x"),
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


async def test_build_model_with_repeat():
    def repeated_ws(times):
        return make_ws(
            pytorch_core.ENV,
            {
                "input": {"title": "Input: tensor"},
                "lin": {"title": "Linear", "output_dim": 8},
                "act": {"title": "Activation", "type": "LeakyReLU"},
                "output": {"title": "Output"},
                "label": {"title": "Input: tensor"},
                "loss": {"title": "MSE loss"},
                "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
                "repeat": {"title": "Repeat", "times": times, "same_weights": False},
            },
            [
                ("input:output", "lin:x"),
                ("lin:output", "act:x"),
                ("act:output", "output:x"),
                ("output:x", "loss:x"),
                ("label:output", "loss:y"),
                ("loss:output", "optim:loss"),
                ("repeat:output", "lin:x"),
                ("act:output", "repeat:input"),
            ],
        )

    # 1 repetition
    m = pytorch_core.build_model(repeated_ws(1))
    assert summarize_layers(m) == "IdLiLeIdIdId"
    assert summarize_connections(m) == "i->S S->l l->a a->E E->o o->o"

    # 2 repetitions
    m = pytorch_core.build_model(repeated_ws(2))
    assert summarize_layers(m) == "IdLiLeIdLiLeIdIdId"
    assert summarize_connections(m) == "i->S S->l l->a a->S S->l l->a a->E E->o o->o"

    # 3 repetitions
    m = pytorch_core.build_model(repeated_ws(3))
    assert summarize_layers(m) == "IdLiLeIdLiLeIdLiLeIdIdId"
    assert summarize_connections(m) == "i->S S->l l->a a->S S->l l->a a->S S->l l->a a->E E->o o->o"


async def test_build_model_with_submodules():
    import torch_geometric.nn as pyg_nn

    @pytorch_core.op("Test submodules")
    def build_submodule(
        x: torch.Tensor, modules: list[torch.nn.Module], single_module: torch.nn.Module
    ):
        return torch.nn.Sequential(*modules, single_module)

    #              / Linear \
    # Input:Tensor --------- Sequential -- Activation -- Output -- Loss -- Optimizer
    #              \ Linear /                           /
    #               \----------------------------------/
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input": {"title": "Input: tensor"},
            "lin1": {"title": "Linear", "output_dim": 8},
            "lin2": {"title": "Linear", "output_dim": 4},
            "seq1": {"title": "Test submodules"},
            "act": {"title": "Activation", "type": "LeakyReLU"},
            "output": {"title": "Output"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input:output", "lin1:x"),
            ("input:output", "lin2:x"),
            ("input:output", "seq1:x"),
            ("input:output", "loss:y"),
            ("lin1:output", "seq1:modules"),
            ("lin1:output", "seq1:single_module"),
            ("lin2:output", "seq1:modules"),
            ("seq1:output", "act:x"),
            ("act:output", "output:x"),
            ("output:x", "loss:x"),
            ("loss:output", "optim:loss"),
        ],
    )
    m = pytorch_core.build_model(ws)
    assert summarize_layers(m) == "SeLeIdId"
    assert len(m.model[0]) == 3 and all(isinstance(layer, pyg_nn.Linear) for layer in m.model[0])
    assert m.model_inputs == [
        "input_output"
    ]  # submodule inputs should not be included in the model inputs


async def test_build_model_with_list_inputs():
    @pytorch_core.op("Test list inputs")
    def build_list_input(x: list[torch.Tensor], y: torch.Tensor):
        return lambda *args: torch.concatenate(args, dim=1)

    ws = make_ws(
        pytorch_core.ENV,
        {
            "input1": {"title": "Input: tensor"},
            "input2": {"title": "Input: tensor"},
            "label": {"title": "Input: tensor"},
            "list_input": {"title": "Test list inputs"},
            "lin": {"title": "Linear", "output_dim": 12},
            "output": {"title": "Output"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input1:output", "list_input:x"),
            ("input1:output", "list_input:y"),
            ("input2:output", "list_input:x"),
            ("list_input:output", "lin:x"),
            ("lin:output", "output:x"),
            ("output:x", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    x1 = torch.rand(100, 4)
    x2 = torch.rand(100, 4)
    y = torch.concatenate([x1, x2, x1], dim=1)
    m = pytorch_core.build_model(ws)
    assert m.model_inputs == ["input1_output", "input2_output"]
    for i in range(200):
        loss = m.train({"input1_output": x1, "input2_output": x2, "label_output": y})
    assert loss < 0.1


async def test_raise_error_on_multiple_edges_to_non_list_input():
    ws = make_ws(
        pytorch_core.ENV,
        {
            "input1": {"title": "Input: tensor"},
            "input2": {"title": "Input: tensor"},
            "lin": {"title": "Linear", "output_dim": 4},
            "output": {"title": "Output"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("input1:output", "lin:x"),
            ("input2:output", "lin:x"),  # Multiple edges to non-list input
            ("lin:output", "output:x"),
            ("output:x", "loss:x"),
            ("output:x", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    with pytest.raises(AssertionError, match="Detected multiple input edges for non-list input"):
        pytorch_core.build_model(ws)


if __name__ == "__main__":
    pytest.main()
