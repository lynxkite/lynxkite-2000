from lynxkite.core import workspace
from lynxkite_graph_analytics import pytorch_model_ops
import torch
import pytest


def make_ws(env, nodes: dict[str, dict], edges: list[tuple[str, str, str, str]]):
    ws = workspace.Workspace(env=env)
    for id, data in nodes.items():
        ws.nodes.append(
            workspace.WorkspaceNode(
                id=id,
                type="basic",
                data=workspace.WorkspaceNodeData(title=data["title"], params=data),
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


async def test_build_model():
    ws = make_ws(
        pytorch_model_ops.ENV,
        {
            "emb": {"title": "Input: embedding"},
            "lin": {"title": "Linear", "output_dim": "same"},
            "act": {"title": "Activation", "type": "Leaky ReLU"},
            "label": {"title": "Input: label"},
            "loss": {"title": "MSE loss"},
            "optim": {"title": "Optimizer", "type": "SGD", "lr": 0.1},
        },
        [
            ("emb:x", "lin:x"),
            ("lin:x", "act:x"),
            ("act:x", "loss:x"),
            ("label:y", "loss:y"),
            ("loss:loss", "optim:loss"),
        ],
    )
    x = torch.rand(100, 4)
    y = x + 1
    m = pytorch_model_ops.build_model(ws, {"emb_x": x, "label_y": y})
    for i in range(1000):
        loss = m.train({"emb_x": x, "label_y": y})
    assert loss < 0.1
    o = m.inference({"emb_x": x[:1]})
    error = torch.nn.functional.mse_loss(o["act_x"], x[:1] + 1)
    assert error < 0.1


if __name__ == "__main__":
    pytest.main()
