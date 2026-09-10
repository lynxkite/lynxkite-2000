**Linear:**

```python
@op("Linear", weights=True)
def linear(x, *, output_dim=1024):
    return pyg_nn.Linear(-1, output_dim)

```
