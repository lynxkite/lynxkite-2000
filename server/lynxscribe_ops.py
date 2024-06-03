'''An example of passive ops. Just using LynxKite to describe the configuration of a complex system.'''
from .ops import passive_op_registration, Parameter as P, MULTI_INPUT

reg = passive_op_registration('LynxScribe')
reg('Scrape documents', params=[P.basic('url', '')])
reg('Conversation logs')
reg('Extract graph', inputs=['input'])
reg('Compute embeddings', inputs=['input'], params=[P.options('method', ['OpenAI', 'graph', 'Yi-34b']), P.basic('dimensions', 1234)])
reg('Vector DB', inputs=[MULTI_INPUT], params=[P.options('backend', ['FAISS', 'ANN', 'HNSW'])])
reg('Chat UI', outputs=[], inputs=['input'])
reg('Chat backend')
reg('WhatsApp')
reg('PII removal', inputs=['input'])
reg('Intent classification', inputs=['input'])
reg('System prompt', params=[P.basic('prompt', 'You are a helpful chatbot.')])
reg('LLM', inputs=[MULTI_INPUT], params=[P.options('backend', ['GPT-4', 'Yi-34b', 'Claude 3 Opus', 'Google Gemini'])])

# From Marton's mock-up.
yi = 'Yi-34B (triton)'
reg('Chat Input', params=[
  P.options('load_mode', ['augmented']),
  P.options('model', [yi]),
  P.options('embedder', ['GritLM-7b (triton)']),
  ])
reg('k-NN Intent Classifier', inputs=['qa_embs', 'rag_graph'], params=[
  P.options('distance', ['cosine', 'euclidean']),
  P.basic('max_dist', 0.3),
  P.basic('k', 3),
  P.options('voting', ['most common', 'weighted']),
  ])
reg('Chroma Graph RAG Loader', inputs=[], params=[
  P.options('location', ['GCP']),
  P.collapsed('bucket', ''),
  P.collapsed('folder', ''),
  P.options('embedder', ['GritLM-7b (triton)']),
  ])
reg('Scenario Builder', inputs=['input'], params=[
  P.collapsed('scenario', ''),
  ])
reg('Graph RAG Answer', inputs=['qa_embs', 'intent', 'rag_graph', 'prompt_dict'], params=[
  P.options('answer_llm', [yi]),
  P.basic('faq_dist', 0.12),
  P.basic('max_dist', 0.25),
  P.basic('ctx_tokens', 2800),
  P.options('distance', ['cosine', 'euclidean']),
  P.collapsed('graph_rag_params', ''),
  ])
reg('Answer Post Processing', inputs=['qa_embs', 'rag_graph'], params=[
  P.options('distance', ['cosine', 'euclidean']),
  P.basic('min_conf', 0.78),
  ])
reg('Chat Output', inputs=['input'], outputs=[])
