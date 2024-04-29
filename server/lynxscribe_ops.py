'''An example of passive ops. Just using LynxKite to describe the configuration of a complex system.'''
from .ops import register_passive_op, Parameter as P

register_passive_op('Scrape documents', inputs={}, params=[P('url', '')])
register_passive_op('Extract graph')
register_passive_op('Compute embeddings')
register_passive_op('Vector DB', params=[P('backend', 'FAISS')])
register_passive_op('Chat UI', outputs={})
register_passive_op('Chat backend', inputs={})
register_passive_op('WhatsApp', inputs={})
register_passive_op('PII removal')
register_passive_op('Intent classification')
register_passive_op('System prompt', inputs={}, params=[P('prompt', 'You are a heplful chatbot.')])
register_passive_op('LLM', inputs={'multi': '*'}, params=[P('model', 'gpt4')])
