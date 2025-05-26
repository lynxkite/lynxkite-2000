import unittest
from lynxkite_lynxscribe import llm_ops  # noqa: F401
from lynxkite.core.executors import one_by_one
from lynxkite.core import ops, workspace


def make_node(id, op, type="basic", **params):
    return workspace.WorkspaceNode(
        id=id,
        type=type,
        position=workspace.Position(x=0, y=0),
        data=workspace.WorkspaceNodeData(title=op, params=params),
    )


def make_input(id):
    return make_node(
        id,
        "Input CSV",
        filename="/Users/danieldarabos/Downloads/aimo-train.csv",
        key="problem",
    )


def make_edge(source, target, targetHandle="input"):
    return workspace.WorkspaceEdge(
        id=f"{source}-{target}",
        source=source,
        target=target,
        sourceHandle="",
        targetHandle=targetHandle,
    )


class LLMOpsTest(unittest.IsolatedAsyncioTestCase):
    async def testExecute(self):
        ws = workspace.Workspace(
            env="LLM logic",
            nodes=[
                make_node(
                    "0",
                    "Input CSV",
                    filename="/Users/danieldarabos/Downloads/aimo-train.csv",
                    key="problem",
                ),
                make_node("1", "View", type="table_view"),
            ],
            edges=[make_edge("0", "1")],
        )
        catalog = ops.CATALOGS[ws.env]
        await one_by_one._execute(ws, catalog)
        # self.assertEqual('', ws.nodes[1].data.display)

    def testStages(self):
        ws = workspace.Workspace(
            env="LLM logic",
            nodes=[
                make_input("in1"),
                make_input("in2"),
                make_input("in3"),
                make_node("rag1", "RAG"),
                make_node("rag2", "RAG"),
                make_node("p1", "Create prompt"),
                make_node("p2", "Create prompt"),
            ],
            edges=[
                make_edge("in1", "rag1", "db"),
                make_edge("in2", "rag1"),
                make_edge("rag1", "p1"),
                make_edge("p1", "rag2", "db"),
                make_edge("in3", "p2"),
                make_edge("p3", "rag2"),
            ],
        )
        catalog = ops.CATALOGS[ws.env]
        stages = one_by_one._get_stages(ws, catalog)
        print(stages)
        # self.assertEqual('', stages)

    def testStagesMultiInput(self):
        ws = workspace.Workspace(
            env="LLM logic",
            nodes=[
                make_node("doc", "Input document"),
                make_node("split", "Split document"),
                make_node("graph", "Build document graph"),
                make_node("chat", "Input chat"),
                make_node("rag", "RAG"),
                make_node("neighbors", "Add neighbors"),
            ],
            edges=[
                make_edge("doc", "split"),
                make_edge("split", "graph"),
                make_edge("split", "rag", "db"),
                make_edge("chat", "rag", "input"),
                make_edge("split", "neighbors", "nodes"),
                make_edge("graph", "neighbors", "edges"),
                make_edge("rag", "neighbors", "item"),
            ],
        )
        catalog = ops.CATALOGS[ws.env]
        stages = one_by_one._get_stages(ws, catalog)
        print(stages)
        # self.assertEqual('', stages)


if __name__ == "__main__":
    unittest.main()
