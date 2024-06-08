import unittest
from . import llm_ops
from . import workspace

def make_node(id, op, type='basic', **params):
  return workspace.WorkspaceNode(
    id=id,
    type=type,
    position=workspace.Position(x=0, y=0),
    data=workspace.WorkspaceNodeData(title=op, params=params),
  )
def make_input(id):
  return make_node(
    id, 'Input',
    filename='/Users/danieldarabos/Downloads/aimo-train.csv',
    key='problem')
def make_edge(source, target, targetHandle='input'):
  return workspace.WorkspaceEdge(
    id=f'{source}-{target}', source=source, target=target, sourceHandle='', targetHandle=targetHandle)

class LLMOpsTest(unittest.TestCase):
  def testExecute(self):
    ws = workspace.Workspace(env='LLM logic', nodes=[
      make_node(
        '0', 'Input',
        filename='/Users/danieldarabos/Downloads/aimo-train.csv',
        key='problem'),
      make_node(
        '1', 'View', type='table_view'),
    ], edges=[
      make_edge('0', '1')
    ])
    llm_ops.execute(ws)
    self.assertEqual('', ws.nodes[1].data.display)

  def testStages(self):
    ws = workspace.Workspace(env='LLM logic', nodes=[
      make_input('in1'), make_input('in2'), make_input('in3'),
      make_node('rag1', 'RAG'), make_node('rag2', 'RAG'),
      make_node('p1', 'Create prompt'), make_node('p2', 'Create prompt'),
    ], edges=[
      make_edge('in1', 'rag1', 'db'), make_edge('in2', 'rag1'),
      make_edge('rag1', 'p1'), make_edge('p1', 'rag2', 'db'),
      make_edge('in3', 'p2'), make_edge('p3', 'rag2'),
    ])
    stages = llm_ops.get_stages(ws)
    self.assertEqual('', stages)

if __name__ == '__main__':
  unittest.main()
