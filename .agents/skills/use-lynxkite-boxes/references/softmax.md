**Softmax:**

```python
@op("Softmax")
def softmax(x, *, dim=1):
    return torch.nn.Softmax(dim=dim)

```
