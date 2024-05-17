'''An example of passive ops. Just using LynxKite to describe the configuration of a complex system.'''
from .ops import register_passive_op as reg, Parameter as P

reg('Scrape documents', inputs={}, params=[P('url', '')])
reg('Conversation logs', inputs={})
reg('Extract graph')
reg('Compute embeddings', params=[P.options('method', ['OpenAI', 'graph', 'Yi-34b']), P('dimensions', 1234)])
reg('Vector DB', inputs={'multi': '*'}, params=[P.options('backend', ['FAISS', 'ANN', 'HNSW'])])
reg('Chat UI', outputs={})
reg('Chat backend', inputs={})
reg('WhatsApp', inputs={})
reg('PII removal')
reg('Intent classification')
reg('System prompt', inputs={}, params=[P('prompt', 'You are a helpful chatbot.')])
reg('LLM', inputs={'multi': '*'}, params=[P.options('backend', ['GPT-4', 'Yi-34b', 'Claude 3 Opus', 'Google Gemini'])])

# From Marton's mock-up.
yi = 'Yi-34B (triton)'
reg('Chat Input', inputs={}, params=[
  P.options('load_mode', ['augmented']),
  P.options('model', [yi]),
  P.options('embedder', ['GritLM-7b (triton)']),
  ])
reg('k-NN Intent Classifier', inputs={'qa_embs': None, 'rag_graph': None}, params=[
  P.options('distance', ['cosine', 'euclidean']),
  P('max_dist', 0.3),
  P('k', 3),
  P.options('voting', ['most common', 'weighted']),
  ])
reg('Chroma Graph RAG Loader', inputs={}, params=[
  P.options('location', ['GCP']),
  P.collapsed('bucket', ''),
  P.collapsed('folder', ''),
  P.options('embedder', ['GritLM-7b (triton)']),
  ])
reg('Scenario Builder', params=[
  P.collapsed('scenario', ''),
  ])
reg('Graph RAG Answer', inputs={'qa_embs': None, 'intent': None, 'rag_graph': None, 'prompt_dict': None}, params=[
  P.options('answer_llm', [yi]),
  P('faq_dist', 0.12),
  P('max_dist', 0.25),
  P('ctx_tokens', 2800),
  P.options('distance', ['cosine', 'euclidean']),
  P.collapsed('graph_rag_params', ''),
  ])
reg('Answer Post Processing', inputs={'qa_embs': None, 'rag_graph': None}, params=[
  P.options('distance', ['cosine', 'euclidean']),
  P('min_conf', 0.78),
  ])
reg('Chat Output', outputs={})
