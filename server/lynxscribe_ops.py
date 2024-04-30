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
