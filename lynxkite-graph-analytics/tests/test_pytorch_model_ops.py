import torch_geometric.nn as pyg_nn
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


def summarize_layers(m: pytorch_core.ModelConfig) -> str:
    assert isinstance(m.model, pyg_nn.Sequential)
    return "".join(str(e)[:2] for e in m.model)


def summarize_connections(m: pytorch_core.ModelConfig) -> str:
    assert isinstance(m.model, pyg_nn.Sequential)
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


if __name__ == "__main__":
    pytest.main()
