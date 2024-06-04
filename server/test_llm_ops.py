import unittest
from . import llm_ops
from . import workspace

class LLMOpsTest(unittest.TestCase):
  def testExecute(self):
    ws = workspace.Workspace(env='LLM logic', nodes=[
      workspace.WorkspaceNode(
        id='0',
        type='basic',
        position=workspace.Position(x=0, y=0),
        data=workspace.WorkspaceNodeData(title='Input', params={
          'filename': '/Users/danieldarabos/Downloads/aimo-train.csv',
          'key': 'problem',
        })),
      workspace.WorkspaceNode(
        id='1',
        type='table_view',
        position=workspace.Position(x=0, y=0),
        data=workspace.WorkspaceNodeData(title='View', params={})),
    ], edges=[
      workspace.WorkspaceEdge(id='0-1', source='0', target='1', sourceHandle='', targetHandle=''),
    ])
    llm_ops.execute(ws)
    self.assertEqual('', ws.nodes[1].data.display)

if __name__ == '__main__':
  unittest.main()
