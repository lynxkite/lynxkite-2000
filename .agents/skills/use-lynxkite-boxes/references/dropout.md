**Dropout:**

```python
@op("Dropout")
def dropout(x, *, p=0.0):
    return torch.nn.Dropout(p)

```
