**Attention:**

```python
@op("Attention", outputs=["outputs", "weights"])
def attention(query, key, value, *, embed_dim=1024, num_heads=1, dropout=0.0):
    return torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

```
