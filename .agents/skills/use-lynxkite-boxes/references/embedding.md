**Embedding:**

```python
@op("Embedding", weights=True)
def embedding(x, *, num_embeddings: int, embedding_dim: int):
    return torch.nn.Embedding(num_embeddings, embedding_dim)

```
