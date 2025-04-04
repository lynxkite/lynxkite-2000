from lynxkite.core import workspace
from lynxkite_graph_analytics import pytorch_model_ops
import torch
import pytest


def make_ws(env, nodes: dict[str, dict], edges: list[tuple[str, str, str, str]]):
    ws = workspace.Workspace(env=env)
    for id, data in nodes.items():
        title = data["title"]
        del data["title"]
        ws.nodes.append(
            workspace.WorkspaceNode(
                id=id,
                type="basic",
                data=workspace.WorkspaceNodeData(title=title, params=data),
                position=workspace.Position(
                    x=data.get("x", 0),
                    y=data.get("y", 0),
                ),
            )
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


def summarize_layers(m: pytorch_model_ops.ModelConfig) -> str:
    return "".join(str(e)[0] for e in m.model)


def summarize_connections(m: pytorch_model_ops.ModelConfig) -> str:
    return " ".join(
        "".join(n[0] for n in c.param_names) + "->" + "".join(n[0] for n in c.return_names)
        for c in m.model._children
    )


async def test_build_model():
    ws = make_ws(
        pytorch_model_ops.ENV,
        {
            "emb": {"title": "Input: tensor"},
            "lin": {"title": "Linear", "output_dim": "same"},
            "act": {"title": "Activation", "type": "Leaky_ReLU"},
            "label": {"title": "Input: tensor"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("emb:output", "lin:x"),
            ("lin:output", "act:x"),
            ("act:output", "loss:x"),
            ("label:output", "loss:y"),
            ("loss:output", "optim:loss"),
        ],
    )
    x = torch.rand(100, 4)
    y = x + 1
    m = pytorch_model_ops.build_model(ws, {"emb_output": x, "label_output": y})
    for i in range(1000):
        loss = m.train({"emb_output": x, "label_output": y})
    assert loss < 0.1
    o = m.inference({"emb_output": x[:1]})
    error = torch.nn.functional.mse_loss(o["act_output"], x[:1] + 1)
    assert error < 0.1


async def test_build_model_with_repeat():
    def repeated_ws(times):
        return make_ws(
            pytorch_model_ops.ENV,
            {
                "emb": {"title": "Input: tensor"},
                "lin": {"title": "Linear", "output_dim": "same"},
                "act": {"title": "Activation", "type": "Leaky_ReLU"},
                "label": {"title": "Input: tensor"},
                "loss": {"title": "MSE loss"},
                "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
                "repeat": {"title": "Repeat", "times": times, "same_weights": False},
            },
            [
                ("emb:output", "lin:x"),
                ("lin:output", "act:x"),
                ("act:output", "loss:x"),
                ("label:output", "loss:y"),
                ("loss:output", "optim:loss"),
                ("repeat:output", "lin:x"),
                ("act:output", "repeat:input"),
            ],
        )

    # 1 repetition
    m = pytorch_model_ops.build_model(repeated_ws(1), {})
    assert summarize_layers(m) == "IL<II"
    assert summarize_connections(m) == "e->S S->l l->a a->E E->E"

    # 2 repetitions
    m = pytorch_model_ops.build_model(repeated_ws(2), {})
    assert summarize_layers(m) == "IL<IL<II"
    assert summarize_connections(m) == "e->S S->l l->a a->S S->l l->a a->E E->E"

    # 3 repetitions
    m = pytorch_model_ops.build_model(repeated_ws(3), {})
    assert summarize_layers(m) == "IL<IL<IL<II"
    assert summarize_connections(m) == "e->S S->l l->a a->S S->l l->a a->S S->l l->a a->E E->E"


if __name__ == "__main__":
    pytest.main()
