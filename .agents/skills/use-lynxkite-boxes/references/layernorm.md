**LayerNorm:**

```python
@op("LayerNorm", outputs=["outputs", "weights"])
def layernorm(x, *, normalized_shape=""):
    normalized_shape = [int(s.strip()) for s in normalized_shape.split(",")]
    return torch.nn.LayerNorm(normalized_shape)

```
